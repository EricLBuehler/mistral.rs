# Ring backend in mistral.rs

Mistral.rs provides a TCP-based ring backend for distributed tensor-parallel inference. This backend is enabled by compiling with the `ring` feature and implements collective operations over a ring topology using TCP sockets.

## Prerequisites
- Build with the `ring` feature enable, **in addition to any others**:
  ```bash
  cargo build --release --features ring
  ```
- Ensure the specified TCP ports are open and reachable between processes.
- The `world_size` must be a power of 2 (2, 4, 8, 16, etc.) for correct operation.

## Configuration
Create one JSON configuration file per process with the following fields:

| Field         | Type     | Description                                                                            |
|---------------|----------|----------------------------------------------------------------------------------------|
| `master_ip`   | string   | Optional. IP address for master node.                         |
| `master_port` | integer  | Optional. Port for master node.                               |
| `port`        | integer  | Local port to bind for incoming connections from the left neighbor.                    |
| `right_port`  | integer  | Port on which the right neighbor is listening (used to connect outgoing to the right). |
| `right_ip`    | string   | Optional. IP address of the right neighbor (defaults to `0.0.0.0`).                     |
| `rank`        | integer  | Rank of this process in `[0..world_size)`.                                            |
| `world_size`  | integer  | Total number of processes in the ring. **Must be a power of 2** (e.g., 2, 4, 8, 16, etc.). |



**This address and port should form a ring topology for each of the nodes.** For example, the last node should point to the first node as its right neighbor.

Although all processes participate in collective communication, Rank 0 acts as the master node. For example, interactive mode or the server runs on Rank 0, while other ranks act as background workers.

Example ring topology:

```text
+---------+         +---------+
| Rank 0  | ----->  | Rank 1  |
| IP: A   |         | IP: B   |
| Port: X |         | Port: Y |
+----+----+         +----+----+
     ^                   |
     |                   v
+----+----+         +----+----+
| Rank 3  | <-----  | Rank 2  |
| IP: D   |         | IP: C   |
| Port: W |         | Port: Z |
+---------+         +---------+
```

Each node connects to its right neighbor by IP and port, and the last node wraps around to the first.

Example for two processes:

- [`ring_0.json`](https://github.com/EricLBuehler/mistral.rs/blob/master/ring_configs/ring_0.json):
  ```json
  {
    "master_ip": "0.0.0.0",
    "master_port": 1234,
    "port": 12345,
    "right_port": 12346,
    "rank": 0,
    "world_size": 2
  }
  ```

- [`ring_0.json`](https://github.com/EricLBuehler/mistral.rs/blob/master/ring_configs/ring_1.json):
  ```json
  {
    "master_ip": "0.0.0.0",
    "master_port": 1234,
    "port": 12346,
    "right_port": 12345,
    "rank": 1,
    "world_size": 2
  }
  ```

### Multi-Machine Example

To run on different machines, update the `right_ip` field in each config to the actual IP address of the neighbor process. For example, if you have two machines with IPs `192.168.1.10` and `192.168.1.11`:

- `ring_0.json` on Machine A (192.168.1.10):
  ```json
  {
    "port": 12345,
    "right_port": 12346,
    "right_ip": "192.168.1.11",
    "rank": 0,
    "world_size": 2
  }
  ```

- `ring_1.json` on Machine B (192.168.1.11):
  ```json
  {
    "port": 12346,
    "right_port": 12345,
    "right_ip": "192.168.1.10",
    "rank": 1,
    "world_size": 2
  }
  ```

Make sure that the specified ports are open and that each machine can reach the other via TCP on those ports.

## Usage
Set the `RING_CONFIG` environment variable to point to the JSON file for each process, then run your application built with the `ring` feature:

```bash
# Process 0 or computer 0
export RING_CONFIG=path/to/ring_0.json
cargo run --release --features ring -- ...

# Process 1 or computer 1
export RING_CONFIG=path/to/ring_1.json
cargo run --release --features ring -- ...
```

The ring backend will automatically handle collective communication for tensor-parallel inference.
