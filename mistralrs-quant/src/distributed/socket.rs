use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    slice,
    sync::{Barrier, Mutex},
    time::{Duration, Instant},
};

use super::{BarrierLike, Id};
use candle_core::Result;

/// The Server maintains persistent connections.
#[derive(Debug)]
pub struct Server {
    // Persistent TCP connections from each node.
    connections: Vec<TcpStream>,
    barrier_all: Barrier,
    barrier_crossnode: Barrier,
}

impl Server {
    /// Binds the listener and then accepts exactly `n_nodes` persistent connections.
    pub fn new<A: ToSocketAddrs>(addr: &A, n_nodes: usize, n_local_ranks: usize) -> Result<Self> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(false)?;
        let start = Instant::now();
        let mut connections = Vec::with_capacity(n_nodes);
        while connections.len() < n_nodes {
            if let Ok((stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs_f32(10.)))?;
                stream.set_write_timeout(Some(Duration::from_secs_f32(10.)))?;

                connections.push(stream);
            }
            if start.elapsed() > Duration::from_secs(10) {
                candle_core::bail!("Worker did not connect to head node due to timeout: over 10s");
            }
        }
        Ok(Self {
            connections,
            barrier_all: Barrier::new(n_local_ranks),
            barrier_crossnode: Barrier::new(n_local_ranks),
        })
    }

    /// Broadcasts the given ID over all persistent connections.
    pub fn broadcast_id(&self, id: &Id) -> Result<()> {
        let body = id.internal();
        // SAFETY: We know the provenance and lifetime of `body` are valid.
        let body_bytes = unsafe { slice::from_raw_parts(body.as_ptr() as *const u8, body.len()) };
        for mut stream in &self.connections {
            stream.write_all(body_bytes)?;
            stream.flush()?;
        }
        Ok(())
    }
}

impl BarrierLike for Server {
    fn wait(&self) -> Result<()> {
        // First, synchronize locally.
        let res = self.barrier_all.wait();

        if res.is_leader() {
            // Leader sends the barrier signal "g" to every node.
            for mut stream in &self.connections {
                stream.write_all(b"g")?;
                stream.flush()?;
            }
            // Now, wait to receive an acknowledgement "a" from every node.
            let mut ack_buf = [0u8; 1];
            for mut stream in &self.connections {
                stream.read_exact(&mut ack_buf)?;
                if &ack_buf != b"a" {
                    candle_core::bail!("Did not get Ack from worker node");
                }
            }
        }

        self.barrier_crossnode.wait();
        Ok(())
    }
}

/// The Client holds its persistent connection inside a Mutex so that its barrier
/// operations can have mutable access to the stream.
#[derive(Debug)]
pub struct Client {
    stream: Mutex<TcpStream>,
    barrier_all: Barrier,
    barrier_crossnode: Barrier,
}

impl Client {
    pub fn new(addr: SocketAddr, n_local_ranks: usize) -> Result<Self> {
        let start = Instant::now();
        loop {
            let stream = TcpStream::connect(addr);
            if let Ok(stream) = stream {
                stream.set_nodelay(true)?;
                stream.set_nonblocking(false)?;

                stream.set_read_timeout(Some(Duration::from_secs_f32(10.)))?;
                stream.set_write_timeout(Some(Duration::from_secs_f32(10.)))?;

                return Ok(Self {
                    stream: Mutex::new(stream),
                    barrier_all: Barrier::new(n_local_ranks),
                    barrier_crossnode: Barrier::new(n_local_ranks),
                });
            }
            if start.elapsed() > Duration::from_secs(10) {
                candle_core::bail!("Failed to connect to head node due to timeout: over 10s");
            }
        }
    }

    /// Receives the broadcasted ID from the persistent stream.
    pub fn receive_id(&self) -> Result<Id> {
        let mut stream = self.stream.lock().unwrap();
        let mut buffer = [0u8; 128];
        stream.read_exact(&mut buffer)?;

        let mut id_bytes: [core::ffi::c_char; 128] = [0; 128];
        for (i, &b) in buffer.iter().enumerate() {
            id_bytes[i] = b as core::ffi::c_char;
        }
        Ok(Id::uninit(id_bytes))
    }
}

impl BarrierLike for Client {
    fn wait(&self) -> Result<()> {
        // Synchronize locally.
        let res = self.barrier_all.wait();

        if res.is_leader() {
            let mut stream = self.stream.lock().unwrap();
            // Read the barrier signal "Go!" from the persistent stream.
            let mut buf = [0u8; 1];
            stream.read_exact(&mut buf)?;
            if &buf != b"g" {
                candle_core::bail!("Did not receive correct barrier signal from head node");
            }
            // Immediately send back an acknowledgement "Ack".
            stream.write_all(b"a")?;
            stream.flush()?;
        }
        // Synchronize again across local ranks.
        self.barrier_crossnode.wait();
        Ok(())
    }
}
