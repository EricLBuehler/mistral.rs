use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    slice,
    sync::{Arc, Barrier, Mutex},
    time::{Duration, Instant},
};

use candle_core::Result;

use super::{BarrierLike, Id};

#[derive(Debug)]
pub struct Server {
    listener: TcpListener,
    n_nodes: usize,
    node_connections: Arc<Mutex<Vec<TcpStream>>>,
    barrier_all: Barrier,
    barrier_crossnode: Barrier,
}

impl Server {
    pub fn new<A: ToSocketAddrs>(addr: &A, n_nodes: usize, n_local_ranks: usize) -> Result<Self> {
        let start = Instant::now();
        let listener = loop {
            match TcpListener::bind(addr) {
                Ok(l) => break l,
                Err(_) if Instant::now().duration_since(start).as_secs_f32() < 10.0 => continue,
                Err(_) => candle_core::bail!(
                    "Worker did not connect to head node due to timeout: over 10s"
                ),
            }
        };
        Ok(Self {
            listener,
            n_nodes,
            node_connections: Arc::new(Mutex::new(Vec::with_capacity(n_nodes))),
            barrier_all: Barrier::new(n_local_ranks),
            barrier_crossnode: Barrier::new(n_local_ranks),
        })
    }

    /// Accept connections from nodes once and store them for reuse.
    pub fn accept_nodes(&self) -> Result<()> {
        let mut connections = self.node_connections.lock().unwrap();
        for stream in self.listener.incoming().take(self.n_nodes) {
            let stream = stream?;
            // Optionally set options (e.g. set_nodelay, timeouts, etc.)
            connections.push(stream);
        }
        Ok(())
    }

    /// Broadcast this ID over all persistent connections.
    pub fn broadcast_id(&self, id: &Id) -> Result<()> {
        let body = id.internal();
        // SAFETY: we know the provenance & lifetime are valid here.
        let body = unsafe { slice::from_raw_parts(body.as_ptr() as *const u8, body.len()) };

        let connections = self.node_connections.lock().unwrap();
        for mut stream in connections.iter() {
            stream.write_all(&body)?;
            stream.flush()?;
        }
        Ok(())
    }
}

impl BarrierLike for Server {
    fn wait(&self) -> Result<()> {
        let res = self.barrier_all.wait();

        if res.is_leader() {
            // Reuse the persistent connections.
            let connections = self.node_connections.lock().unwrap();
            for mut stream in connections.iter() {
                // Send go ahead response over the already open connection.
                stream.write_all(b"Go!")?;
                stream.flush()?;
            }
        }
        self.barrier_crossnode.wait();

        Ok(())
    }
}

#[derive(Debug)]
pub struct Client {
    connection: Arc<Mutex<TcpStream>>,
    barrier_all: Barrier,
    barrier_crossnode: Barrier,
}

impl Client {
    pub fn new(addr: SocketAddr, n_local_ranks: usize) -> Result<Self> {
        // Connect once with a 10s timeout.
        let connection = Self::connect_once(&addr, Duration::from_secs(10))?;
        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
            barrier_all: Barrier::new(n_local_ranks),
            barrier_crossnode: Barrier::new(n_local_ranks),
        })
    }

    fn connect_once(addr: &SocketAddr, timeout: Duration) -> Result<TcpStream> {
        let start = Instant::now();
        loop {
            match TcpStream::connect(addr) {
                Ok(stream) => return Ok(stream),
                Err(_) if Instant::now().duration_since(start) < timeout => continue,
                Err(_) => candle_core::bail!("Client connect timeout: over {timeout:?}"),
            }
        }
    }

    /// Read the ID from the persistent connection.
    pub fn receive_id(&mut self) -> Result<Id> {
        let mut internal = [0u8; 128];
        self.connection.lock().unwrap().read_exact(&mut internal)?;
        let body_as_i8: &[i8] =
            unsafe { std::slice::from_raw_parts(internal.as_ptr() as *const i8, internal.len()) };

        assert_eq!(body_as_i8.len(), 128);
        let mut uninit = [0i8; 128];
        for (i, x) in body_as_i8.into_iter().enumerate() {
            uninit[i] = *x;
        }
        Ok(Id::uninit(uninit))
    }
}

impl BarrierLike for Client {
    fn wait(&self) -> Result<()> {
        let res = self.barrier_all.wait();

        if res.is_leader() {
            // Read the go-ahead message from the persistent connection.
            let mut out = [0u8; 128];
            let n = self.connection.lock().unwrap().read(&mut out)?;
            assert_ne!(n, 0);
        }

        self.barrier_crossnode.wait();

        Ok(())
    }
}
