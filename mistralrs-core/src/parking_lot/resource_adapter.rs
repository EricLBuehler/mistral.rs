//! Resource cost calculation for LLM inference jobs.
//!
//! This module provides the mapping between LLM-specific resources
//! (KV-cache blocks, GPU memory) and the generic resource units
//! used by the prometheus_parking_lot scheduler.

use prometheus_parking_lot::util::serde::{ResourceCost, ResourceKind};

/// Default block size for KV-cache (tokens per block)
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Resource adapter for mapping inference job requirements to resource costs.
#[derive(Debug, Clone)]
pub struct ResourceAdapter {
    /// Block size for KV-cache (tokens per block)
    block_size: usize,

    /// Total number of GPU blocks available
    total_gpu_blocks: usize,

    /// Memory per block in bytes (for VRAM-based costing)
    bytes_per_block: usize,
}

impl ResourceAdapter {
    /// Create a new resource adapter with explicit parameters.
    #[must_use]
    pub fn new(block_size: usize, total_gpu_blocks: usize, bytes_per_block: usize) -> Self {
        Self {
            block_size,
            total_gpu_blocks,
            bytes_per_block,
        }
    }

    /// Calculate the number of blocks needed for a given number of tokens.
    #[must_use]
    pub fn tokens_to_blocks(&self, num_tokens: usize) -> usize {
        // Round up to nearest block
        (num_tokens + self.block_size - 1) / self.block_size
    }

    /// Calculate resource cost for an inference job.
    ///
    /// The cost is based on the number of KV-cache blocks needed for the
    /// prompt plus expected generation length.
    #[must_use]
    pub fn calculate_cost(&self, prompt_len: usize, max_new_tokens: usize) -> ResourceCost {
        // Total tokens = prompt + expected generation
        let total_tokens = prompt_len + max_new_tokens;
        let blocks_needed = self.tokens_to_blocks(total_tokens);

        ResourceCost {
            kind: ResourceKind::GpuVram,
            units: blocks_needed as u32,
        }
    }

    /// Calculate resource cost in VRAM bytes.
    #[must_use]
    pub fn calculate_vram_cost(&self, prompt_len: usize, max_new_tokens: usize) -> ResourceCost {
        let blocks_needed = self.tokens_to_blocks(prompt_len + max_new_tokens);
        let bytes_needed = blocks_needed * self.bytes_per_block;

        ResourceCost {
            kind: ResourceKind::GpuVram,
            units: bytes_needed as u32,
        }
    }

    /// Get the maximum resource units available (total GPU blocks).
    #[must_use]
    pub fn max_units(&self) -> usize {
        self.total_gpu_blocks
    }

    /// Get the block size.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Calculate available tokens based on free blocks.
    #[must_use]
    pub fn blocks_to_tokens(&self, num_blocks: usize) -> usize {
        num_blocks * self.block_size
    }
}

impl Default for ResourceAdapter {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            total_gpu_blocks: 1024,                      // Default fallback
            bytes_per_block: DEFAULT_BLOCK_SIZE * 2 * 2, // Assuming f16, k+v
        }
    }
}

/// Calculate resource cost for an inference job (convenience function).
///
/// # Arguments
///
/// * `prompt_len` - Number of tokens in the prompt
/// * `max_new_tokens` - Maximum number of tokens to generate
///
/// # Returns
///
/// Resource cost in GPU blocks
#[must_use]
pub fn calculate_resource_cost(prompt_len: usize, max_new_tokens: usize) -> ResourceCost {
    let adapter = ResourceAdapter::default();
    adapter.calculate_cost(prompt_len, max_new_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_to_blocks() {
        let adapter = ResourceAdapter::new(16, 1024, 64);

        assert_eq!(adapter.tokens_to_blocks(0), 0);
        assert_eq!(adapter.tokens_to_blocks(1), 1);
        assert_eq!(adapter.tokens_to_blocks(16), 1);
        assert_eq!(adapter.tokens_to_blocks(17), 2);
        assert_eq!(adapter.tokens_to_blocks(32), 2);
        assert_eq!(adapter.tokens_to_blocks(33), 3);
    }

    #[test]
    fn test_calculate_cost() {
        let adapter = ResourceAdapter::new(16, 1024, 64);

        // 100 prompt tokens + 50 max new = 150 total
        // 150 / 16 = 9.375, rounds up to 10 blocks
        let cost = adapter.calculate_cost(100, 50);
        assert_eq!(cost.units, 10);
        assert!(matches!(cost.kind, ResourceKind::GpuVram));
    }

    #[test]
    fn test_blocks_to_tokens() {
        let adapter = ResourceAdapter::new(16, 1024, 64);

        assert_eq!(adapter.blocks_to_tokens(0), 0);
        assert_eq!(adapter.blocks_to_tokens(1), 16);
        assert_eq!(adapter.blocks_to_tokens(10), 160);
    }
}
