use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream, ToSocketAddrs},
    slice,
    time::Instant,
};

use candle_core::Result;

use super::Id;

pub struct Server {
    listener: TcpListener,
    n_nodes: usize,
}

impl Server {
    pub fn new<A: ToSocketAddrs>(addr: &A, n_nodes: usize) -> Result<Self> {
        let start = Instant::now();
        loop {
            let listener = TcpListener::bind(addr);
            if let Ok(listener) = listener {
                return Ok(Self { listener, n_nodes });
            }
            if Instant::now().duration_since(start).as_secs_f32() >= 10. {
                candle_core::bail!("Client connect timeout: over 10s")
            }
        }
    }

    /// Broadcast this ID to the specified number of nodes (different from ranks)
    pub fn broadcast_id(&self, id: &Id) -> Result<()> {
        for stream in self.listener.incoming().take(self.n_nodes) {
            let mut stream = stream?;

            let body = id.internal();

            // SAFETY: we know the provenance & lifetime are valid here.
            let body = unsafe { slice::from_raw_parts(body.as_ptr() as *const u8, body.len()) };

            // Build and send the HTTP response.
            stream.write_all(&body)?;
            stream.flush()?;
        }

        Ok(())
    }
}

pub struct Client {
    stream: TcpStream,
}

impl Client {
    /// Creates a stream, with a connection timeout of 10 seconds.
    pub fn new<A: ToSocketAddrs>(addr: &A) -> Result<Self> {
        let start = Instant::now();
        loop {
            let stream = TcpStream::connect(addr);
            if let Ok(stream) = stream {
                return Ok(Self { stream });
            }
            if Instant::now().duration_since(start).as_secs_f32() >= 10. {
                candle_core::bail!("Client connect timeout: over 10s")
            }
        }
    }

    pub fn recieve_id(&mut self) -> Result<Id> {
        // Read data into a buffer, we node there are 128.
        let mut internal = [0u8; 128];
        let n = self.stream.read(&mut internal)?;
        assert_eq!(n, internal.len());

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
