use std::{ops::Range, sync::Arc};

pub(crate) trait BlockTableRows {
    fn len(&self) -> usize;

    fn row(&self, index: usize) -> &[usize];
}

impl BlockTableRows for [Vec<usize>] {
    fn len(&self) -> usize {
        <[Vec<usize>]>::len(self)
    }

    fn row(&self, index: usize) -> &[usize] {
        &self[index]
    }
}

impl BlockTableRows for Vec<Vec<usize>> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn row(&self, index: usize) -> &[usize] {
        &self[index]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BlockTableSnapshot {
    tables: Vec<Arc<[usize]>>,
    row_table_indices: Vec<usize>,
}

impl BlockTableSnapshot {
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn from_owned_sequence_tables(
        tables: Vec<Vec<usize>>,
        rows_per_sequence: usize,
    ) -> Self {
        Self::from_sequence_tables(
            tables.into_iter().map(Arc::<[usize]>::from).collect(),
            rows_per_sequence,
        )
    }

    pub(crate) fn from_sequence_tables(
        tables: Vec<Arc<[usize]>>,
        rows_per_sequence: usize,
    ) -> Self {
        let row_table_indices = (0..tables.len())
            .flat_map(|table_idx| std::iter::repeat_n(table_idx, rows_per_sequence))
            .collect();
        Self {
            tables,
            row_table_indices,
        }
    }

    pub(crate) fn from_mapped_tables(
        tables: Vec<Arc<[usize]>>,
        row_table_indices: Vec<usize>,
    ) -> Self {
        debug_assert!(row_table_indices.iter().all(|&idx| idx < tables.len()));
        Self {
            tables,
            row_table_indices,
        }
    }

    pub(crate) fn row_table_index(&self, row: usize) -> usize {
        self.row_table_indices[row]
    }

    pub(crate) fn table(&self, table_idx: usize) -> &[usize] {
        &self.tables[table_idx]
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn table_arc(&self, table_idx: usize) -> Arc<[usize]> {
        self.tables[table_idx].clone()
    }

    pub(crate) fn push_rows_for_table(&mut self, table_idx: usize, count: usize) {
        debug_assert!(table_idx < self.tables.len());
        self.row_table_indices
            .extend(std::iter::repeat_n(table_idx, count));
    }

    #[cfg(test)]
    pub(crate) fn unique_table_count(&self) -> usize {
        self.tables.len()
    }
}

impl BlockTableRows for BlockTableSnapshot {
    fn len(&self) -> usize {
        self.row_table_indices.len()
    }

    fn row(&self, index: usize) -> &[usize] {
        self.table(self.row_table_index(index))
    }
}

pub(crate) struct BlockTableRanges<'a> {
    rows: &'a BlockTableSnapshot,
    ranges: Vec<Range<usize>>,
}

impl<'a> BlockTableRanges<'a> {
    pub(crate) fn new(rows: &'a BlockTableSnapshot, ranges: Vec<Range<usize>>) -> Self {
        debug_assert_eq!(rows.len(), ranges.len());
        Self { rows, ranges }
    }
}

impl BlockTableRows for BlockTableRanges<'_> {
    fn len(&self) -> usize {
        self.ranges.len()
    }

    fn row(&self, index: usize) -> &[usize] {
        self.rows
            .row(index)
            .get(self.ranges[index].clone())
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_tables_share_storage_across_query_rows() {
        let first: Arc<[usize]> = vec![1, 2, 3].into();
        let second: Arc<[usize]> = vec![7, 8].into();
        let rows = BlockTableSnapshot::from_sequence_tables(vec![first.clone(), second.clone()], 3);

        assert_eq!(rows.len(), 6);
        assert_eq!(rows.unique_table_count(), 2);
        assert_eq!(rows.row(0), &[1, 2, 3]);
        assert_eq!(rows.row(2), &[1, 2, 3]);
        assert_eq!(rows.row(3), &[7, 8]);
        assert!(Arc::ptr_eq(&first, &rows.table_arc(0)));
        assert!(Arc::ptr_eq(&second, &rows.table_arc(1)));
    }

    #[test]
    fn ranges_materialize_exact_row_slices() {
        let rows = BlockTableSnapshot::from_sequence_tables(
            vec![vec![1, 2, 3, 4].into(), vec![7, 8, 9].into()],
            2,
        );
        let view = BlockTableRanges::new(&rows, vec![1..3, 2..4, 0..2, 1..3]);

        assert_eq!(view.row(0), &[2, 3]);
        assert_eq!(view.row(1), &[3, 4]);
        assert_eq!(view.row(2), &[7, 8]);
        assert_eq!(view.row(3), &[8, 9]);
    }
}
