use super::{CeilAsUSize, ImplicitFunction, Mesh};
use Plane;
use alga::general::Real;
use bbox::BoundingBox;
use bitset::BitSet;
use cell_configs::CELL_CONFIGS;
use na;
use num_traits::Float;
use qef;
use rand;
use rayon::prelude::*;
use std::{error, fmt};
use std::cell::{Cell, RefCell};
use std::cmp;
use std::collections::{BTreeSet, HashMap};
use vertex_index::{neg_offset, offset, Index, VarIndex, VertexIndex, EDGES_ON_FACE};

// How accurately find zero crossings.
const PRECISION: f32 = 0.05;

//  Edge indexes
//
//      +-------9-------+
//     /|              /|
//    7 |            10 |              ^
//   /  8            /  11            /
//  +-------6-------+   |     ^    higher indexes in y
//  |   |           |   |     |     /
//  |   +-------3---|---+     |    /
//  2  /            5  /  higher indexes
//  | 1             | 4      in z
//  |/              |/        |/
//  o-------0-------+         +-- higher indexes in x ---->
//
// Point o is the reference point of the current cell.
// All edges go from lower indexes to higher indexes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Edge {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
    G = 6,
    H = 7,
    I = 8,
    J = 9,
    K = 10,
    L = 11,
}

impl Edge {
    pub fn from_usize(e: usize) -> Edge {
        match e {
            0 => Edge::A,
            1 => Edge::B,
            2 => Edge::C,
            3 => Edge::D,
            4 => Edge::E,
            5 => Edge::F,
            6 => Edge::G,
            7 => Edge::H,
            8 => Edge::I,
            9 => Edge::J,
            10 => Edge::K,
            11 => Edge::L,
            _ => panic!("Not edge for {:?}", e),
        }
    }
    pub fn base(&self) -> Edge {
        Edge::from_usize(*self as usize % 3)
    }
}

// Cell offsets of edges
const EDGE_OFFSET: [Index; 12] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
];

// Quad definition for edges 0-2.
const QUADS: [[Edge; 4]; 3] = [
    [Edge::A, Edge::G, Edge::J, Edge::D],
    [Edge::B, Edge::E, Edge::K, Edge::H],
    [Edge::C, Edge::I, Edge::L, Edge::F],
];

lazy_static! {
    static ref OUTSIDE_EDGES_PER_CORNER: [BitSet; 8] = [BitSet::from_3bits(0, 1, 2),
                                                        BitSet::from_3bits(0, 4, 5),
                                                        BitSet::from_3bits(1, 3, 8),
                                                        BitSet::from_3bits(3, 4, 11),
                                                        BitSet::from_3bits(2, 6, 7),
                                                        BitSet::from_3bits(5, 6, 10),
                                                        BitSet::from_3bits(7, 8, 9),
                                                        BitSet::from_3bits(9, 10, 11)];
}

#[derive(Debug)]
pub enum DualContouringError {
    HitZero(String),
}

impl error::Error for DualContouringError {
    fn description(&self) -> &str {
        match self {
            &DualContouringError::HitZero(_) => "Hit zero value during grid sampling.",
        }
    }
}

impl fmt::Display for DualContouringError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &DualContouringError::HitZero(ref s) => write!(f, "Hit zero value for {}", s),
        }
    }
}

// A vertex of the mesh. This can be either a primary vertex of the sampled mesh or a vertex
// generated by joining multiple vertices in the octree.
#[derive(Debug)]
pub struct Vertex<S: Real> {
    index: Index,
    qef: RefCell<qef::Qef<S>>,
    neighbors: [Vec<VarIndex>; 6],
    parent: Cell<Option<usize>>,
    children: Vec<usize>,
    // Index of this vertex in the final mesh.
    mesh_index: Cell<Option<usize>>,
    edge_intersections: [u32; 12],
    euler_characteristic: i32,
}

impl<S: Real> Clone for Vertex<S> {
    fn clone(&self) -> Vertex<S> {
        Vertex {
            index: self.index,
            qef: self.qef.clone(),
            neighbors: [
                self.neighbors[0].clone(),
                self.neighbors[1].clone(),
                self.neighbors[2].clone(),
                self.neighbors[3].clone(),
                self.neighbors[4].clone(),
                self.neighbors[5].clone(),
            ],
            parent: self.parent.clone(),
            children: self.children.clone(),
            mesh_index: self.mesh_index.clone(),
            edge_intersections: self.edge_intersections,
            euler_characteristic: self.euler_characteristic,
        }
    }
}

impl<S: Real> Vertex<S> {
    fn is_2manifold(&self) -> bool {
        if self.euler_characteristic != 1 {
            return false;
        }
        for edges_on_face in EDGES_ON_FACE.iter() {
            let mut sum = 0;
            for edge in edges_on_face.into_iter() {
                sum += self.edge_intersections[edge];
            }
            if sum != 0 && sum != 2 {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EdgeIndex {
    edge: Edge,
    index: Index,
}

impl EdgeIndex {
    pub fn base(&self) -> EdgeIndex {
        EdgeIndex {
            edge: self.edge.base(),
            index: offset(self.index, EDGE_OFFSET[self.edge as usize]),
        }
    }
}

pub struct ManifoldDualContouring<'a, S: Real + CeilAsUSize + From<f32>> {
    impl_: ManifoldDualContouringImpl<'a, S>,
}
impl<'a, S: Real + CeilAsUSize + From<f32>> ManifoldDualContouring<'a, S> {
    // Constructor
    // f: implicit function to tessellate
    // res: resolution
    // relative_error: acceptable error threshold when simplifying the mesh.
    pub fn new(f: &'a ImplicitFunction<S>, res: S, relative_error: S) -> ManifoldDualContouring<S> {
        ManifoldDualContouring {
            impl_: ManifoldDualContouringImpl::new(f, res, relative_error),
        }
    }
    pub fn tessellate(&mut self) -> Option<Mesh<S>> {
        self.impl_.tessellate()
    }
}

#[derive(Clone)]
pub struct ManifoldDualContouringImpl<'a, S: Real> {
    function: &'a ImplicitFunction<S>,
    origin: na::Point3<S>,
    dim: [usize; 3],
    mesh: RefCell<Mesh<S>>,
    res: S,
    error: S,
    value_grid: HashMap<Index, S>,
    pub edge_grid: RefCell<HashMap<EdgeIndex, Plane<S>>>,
    // The Vertex Octtree. vertex_octtree[0] stores the leaf vertices. vertex_octtree[1] the next
    // layer and so on. vertex_octtree.len() is the depth of the octtree.
    pub vertex_octtree: Vec<Vec<Vertex<S>>>,
    // Map from VertexIndex to vertex_octtree[0]
    pub vertex_index_map: HashMap<VertexIndex, usize>,
}

// Returns the next largest power of 2
fn pow2roundup(x: usize) -> usize {
    let mut x = x;
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}


// Returns a BitSet containing all egdes connected to "edge" in this cell.
fn get_connected_edges(edge: Edge, cell: BitSet) -> BitSet {
    for &edge_set in CELL_CONFIGS[cell.as_u32() as usize].iter() {
        if edge_set.get(edge as usize) {
            return edge_set;
        }
    }
    panic!("Did not find edge_set for {:?} and {:?}", edge, cell);
}

// Returns all BitSets containing  egdes connected to one of edge_set in this cell.
fn get_connected_edges_from_edge_set(edge_set: BitSet, cell: BitSet) -> Vec<BitSet> {
    let mut result = Vec::new();
    for &cell_edge_set in CELL_CONFIGS[cell.as_u32() as usize].iter() {
        if !cell_edge_set.intersect(edge_set).empty() {
            result.push(cell_edge_set);
        }
    }
    debug_assert!(
        result
            .iter()
            .fold(BitSet::zero(), |sum, x| sum.merge(*x))
            .intersect(edge_set) == edge_set,
        "result: {:?} does not contain all edges from egde_set: {:?}",
        result,
        edge_set
    );
    result
}

fn half_index(input: &Index) -> Index {
    [input[0] / 2, input[1] / 2, input[2] / 2]
}

// Will add the following vertices to neighbors:
// All vertices in the same octtree subcell as start and connected to start.
fn add_connected_vertices_in_subcell<S: Real>(
    base: &Vec<Vertex<S>>,
    start: &Vertex<S>,
    neigbors: &mut BTreeSet<usize>,
) {
    let parent_index = half_index(&start.index);
    for neighbor_index_vector in start.neighbors.iter() {
        for neighbor_index in neighbor_index_vector.iter() {
            match neighbor_index {
                &VarIndex::Index(vi) => {
                    let ref neighbor = base[vi];
                    if half_index(&neighbor.index) == parent_index {
                        if neigbors.insert(vi) {
                            add_connected_vertices_in_subcell(base, &base[vi], neigbors);
                        }
                    }
                }
                &VarIndex::VertexIndex(vi) => {
                    panic!("unexpected VertexIndex {:?}", vi);
                }
            }
        }
    }
}

fn add_child_to_parent<S: Real + Float + From<f32>>(child: &Vertex<S>, parent: &mut Vertex<S>) {
    parent.qef.borrow_mut().merge(&*child.qef.borrow());
    for dim in 0..3 {
        let relevant_neighbor = dim * 2 + (child.index[dim] & 1);
        for neighbor in child.neighbors[relevant_neighbor].iter() {
            if !parent.neighbors[relevant_neighbor].contains(neighbor) {
                parent.neighbors[relevant_neighbor].push(*neighbor);
            }
        }
    }
}

fn subsample_euler_characteristics<S: Real>(
    children: &BTreeSet<usize>,
    vertices: &Vec<Vertex<S>>,
) -> ([u32; 12], i32) {
    let mut intersections = [0u32; 12];
    let mut euler = 0i32;
    let mut inner_sum = 0;
    for vertex in children.iter().map(|i| &vertices[*i]) {
        let i = vertex.index;
        let corner_index = (i[2] & 1) << 2 | (i[1] & 1) << 1 | (i[0] & 1);
        let outside_edges = OUTSIDE_EDGES_PER_CORNER[corner_index];
        for i in 0..12 {
            if outside_edges.get(i) {
                intersections[i] += vertex.edge_intersections[i];
            } else {
                inner_sum += vertex.edge_intersections[i];
            }
        }
        euler += vertex.euler_characteristic;
    }
    debug_assert_eq!(
        inner_sum % 4,
        0,
        "inner_sum {} is not divisible by 4.",
        inner_sum
    );
    euler -= inner_sum as i32 / 4;
    (intersections, euler)
}

pub fn subsample_octtree<S: Real + Float + From<f32>>(base: &Vec<Vertex<S>>) -> Vec<Vertex<S>> {
    let mut result = Vec::new();
    for (i, vertex) in base.iter().enumerate() {
        if vertex.parent.get() == None {
            let mut neighbor_set = BTreeSet::new();
            neighbor_set.insert(i);
            add_connected_vertices_in_subcell(base, vertex, &mut neighbor_set);
            let (intersections, euler) = subsample_euler_characteristics(&neighbor_set, base);
            let mut parent = Vertex {
                index: half_index(&vertex.index),
                qef: RefCell::new(qef::Qef::new(&[], BoundingBox::neg_infinity())),
                neighbors: [
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ],
                parent: Cell::new(None),
                children: Vec::new(),
                mesh_index: Cell::new(None),
                edge_intersections: intersections,
                euler_characteristic: euler,
            };
            for &neighbor_index in neighbor_set.iter() {
                let child = &base[neighbor_index];
                debug_assert!(
                    child.parent.get() == None,
                    "child #{:?} already has parent #{:?}",
                    neighbor_index,
                    child.parent.get().unwrap()
                );
                debug_assert!(!parent.children.contains(&neighbor_index));
                parent.children.push(neighbor_index);
                add_child_to_parent(child, &mut parent);
                child.parent.set(Some(result.len()));
            }
            result.push(parent);
        }
    }
    for vertex in result.iter_mut() {
        for neighbor_vec in vertex.neighbors.iter_mut() {
            for neighbor in neighbor_vec.iter_mut() {
                match neighbor {
                    &mut VarIndex::VertexIndex(_) => {
                        panic!("unexpected VertexIndex in normal node.")
                    }
                    &mut VarIndex::Index(i) => {
                        *neighbor = VarIndex::Index(base[i].parent.get().unwrap())
                    }
                }
            }
        }
    }
    result
}

struct Timer {
    t: ::time::Tm,
}

impl Timer {
    fn new() -> Timer {
        Timer { t: ::time::now() }
    }
    fn elapsed(&mut self) -> ::time::Duration {
        let now = ::time::now();
        let result = now - self.t;
        self.t = now;
        result
    }
}

impl<'a, S: From<f32> + Real + Float + CeilAsUSize> ManifoldDualContouringImpl<'a, S> {
    // Constructor
    // f: function to tessellate
    // res: resolution
    // relative_error: acceptable error threshold when simplifying the mesh.
    pub fn new(
        f: &'a ImplicitFunction<S>,
        res: S,
        relative_error: S,
    ) -> ManifoldDualContouringImpl<'a, S> {
        let _1: S = From::from(1f32);
        let mut bbox = f.bbox().clone();
        bbox.dilate(_1 + res * From::from(1.1f32));
        ManifoldDualContouringImpl {
            function: f,
            origin: bbox.min,
            dim: [
                (bbox.dim()[0] / res).ceil_as_usize(),
                (bbox.dim()[1] / res).ceil_as_usize(),
                (bbox.dim()[2] / res).ceil_as_usize(),
            ],
            mesh: RefCell::new(Mesh {
                vertices: Vec::new(),
                faces: Vec::new(),
            }),
            res: res,
            error: res * relative_error,
            value_grid: HashMap::new(),
            edge_grid: RefCell::new(HashMap::new()),
            vertex_octtree: Vec::new(),
            vertex_index_map: HashMap::new(),
        }
    }
    pub fn tessellate(&mut self) -> Option<Mesh<S>> {
        println!(
            "ManifoldDualContouringImpl: res: {:} {:?}",
            self.res,
            self.function.bbox()
        );
        loop {
            match self.try_tessellate() {
                Ok(mesh) => return Some(mesh),
                // Tessellation failed, b/c the value in one of the grid cells was exactly zero.
                // Retry with some random padding and hope for the best.
                Err(e) => {
                    let padding = na::Vector3::new(
                        -self.res / From::from(10. + rand::random::<f32>().abs()),
                        -self.res / From::from(10. + rand::random::<f32>().abs()),
                        -self.res / From::from(10. + rand::random::<f32>().abs()),
                    );
                    println!("Error: {:?}. moving by {:?} and retrying.", e, padding);
                    self.origin += padding;
                    self.value_grid.clear();
                    self.mesh.borrow_mut().vertices.clear();
                    self.mesh.borrow_mut().faces.clear();
                    self.vertex_octtree.clear();
                    self.vertex_index_map.clear();
                }
            }
        }
    }

    pub fn tessellation_step1(&mut self) -> Option<DualContouringError> {
        let maxdim = cmp::max(self.dim[0], cmp::max(self.dim[1], self.dim[2]));
        let origin = self.origin;
        let origin_value = self.function.value(origin);

        return self.sample_value_grid([0, 0, 0], origin, pow2roundup(maxdim), origin_value);
    }

    // This method does the main work of tessellation.
    // It may fail, if the value in one of the grid cells yields exactly zero.
    fn try_tessellate(&mut self) -> Result<Mesh<S>, DualContouringError> {
        let mut t = Timer::new();
        if let Some(e) = self.tessellation_step1() {
            return Err(e);
        }
        let total_cells = self.dim[0] * self.dim[1] * self.dim[2];
        println!(
            "generated value_grid with {:} % of {:} cells in {:}.",
            (100 * self.value_grid.len()) as f64 / total_cells as f64,
            total_cells,
            t.elapsed()
        );

        self.compact_value_grid();
        println!(
            "compacted value_grid, now {:} % of {:} cells in {:}.",
            (100 * self.value_grid.len()) as f64 / total_cells as f64,
            total_cells,
            t.elapsed()
        );

        self.generate_edge_grid();

        println!(
            "generated edge_grid with {} edges: {:}",
            self.edge_grid.borrow().len(),
            t.elapsed()
        );

        let (leafs, index_map) = self.generate_leaf_vertices();
        self.vertex_index_map = index_map;
        self.vertex_octtree.push(leafs);

        println!(
            "generated {:?} leaf vertices: {:}",
            self.vertex_octtree[0].len(),
            t.elapsed()
        );

        loop {
            let next = subsample_octtree(self.vertex_octtree.last().unwrap());
            if next.len() == self.vertex_octtree.last().unwrap().len() {
                break;
            }
            self.vertex_octtree.push(next);
        }
        println!("subsampled octtree {:}", t.elapsed());

        let num_qefs_solved = self.solve_qefs();

        println!("solved {} qefs: {:}", num_qefs_solved, t.elapsed());

        for edge_index in self.edge_grid.borrow().keys() {
            self.compute_quad(*edge_index);
        }
        println!("generated quads: {:}", t.elapsed());

        println!(
            "computed mesh with {:?} faces.",
            self.mesh.borrow().faces.len()
        );

        Ok(self.mesh.borrow().clone())
    }

    fn sample_value_grid(
        &mut self,
        idx: Index,
        pos: na::Point3<S>,
        size: usize,
        val: S,
    ) -> Option<DualContouringError> {
        debug_assert!(size > 1);
        let mut midx = idx;
        let size = size / 2;
        let size_s: S = From::from(size as f32);
        let vpos = [
            pos,
            pos + na::Vector3::new(self.res, self.res, self.res) * size_s,
        ];
        let sub_cube_diagonal = size_s * self.res * Float::sqrt(From::from(3f32));

        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let mpos = na::Point3::new(vpos[x].x, vpos[y].y, vpos[z].z);
                    let value = if midx == idx {
                        val
                    } else {
                        self.function.value(mpos)
                    };

                    if value == From::from(0f32) {
                        return Some(DualContouringError::HitZero(format!("{}", mpos)));
                    }

                    if size > 1 && Float::abs(value) <= sub_cube_diagonal {
                        if let Some(e) = self.sample_value_grid(midx, mpos, size, value) {
                            return Some(e);
                        }
                    } else {
                        self.value_grid.insert(midx, value);
                    }
                    midx[0] += size;
                }
                midx[0] -= 2 * size;
                midx[1] += size;
            }
            midx[1] -= 2 * size;
            midx[2] += size;
        }
        None
    }

    // Delete all values from value grid that do not have a value of opposing signum in any
    // neighboring index.
    // This might reduces memory usage by ~10x.
    pub fn compact_value_grid(&mut self) {
        // Collect all indexes to remove.
        let value_grid = &mut self.value_grid;
        let keys_to_remove: Vec<_> = value_grid
            .par_iter()
            .filter(|&(idx, &v)| {
                for z in 0..3 {
                    for y in 0..3 {
                        for x in 0..3 {
                            let mut adjacent_idx = idx.clone();
                            adjacent_idx[0] += x - 1;
                            adjacent_idx[1] += y - 1;
                            adjacent_idx[2] += z - 1;
                            if let Some(&adjacent_value) = value_grid.get(&adjacent_idx) {
                                if Float::signum(v) != Float::signum(adjacent_value) {
                                    // Don't collect indexes with
                                    // opposing signum.
                                    return false;
                                }
                            }
                        }
                    }
                }
                return true;
            })
            .map(|(k, _)| k.clone())
            .collect();
        for k in keys_to_remove {
            value_grid.remove(&k);
        }
        value_grid.shrink_to_fit();
    }

    // Store crossing positions of edges in edge_grid
    pub fn generate_edge_grid(&mut self) {
        let mut edge_grid = self.edge_grid.borrow_mut();
        for (&point_idx, &point_value) in &self.value_grid {
            for &edge in [Edge::A, Edge::B, Edge::C].iter() {
                let mut adjacent_idx = point_idx.clone();
                adjacent_idx[edge as usize] += 1;
                if let Some(&adjacent_value) = self.value_grid.get(&adjacent_idx) {
                    let point_pos = self.origin
                        + na::Vector3::new(
                            From::from(point_idx[0] as f32),
                            From::from(point_idx[1] as f32),
                            From::from(point_idx[2] as f32),
                        ) * self.res;
                    let mut adjacent_pos = point_pos;
                    adjacent_pos[edge as usize] += self.res;
                    if let Some(plane) =
                        self.find_zero(point_pos, point_value, adjacent_pos, adjacent_value)
                    {
                        edge_grid.insert(
                            EdgeIndex {
                                edge: edge,
                                index: point_idx,
                            },
                            plane,
                        );
                    }
                }
            }
        }
    }

    // Solves QEFs in vertex stack, starting at the highest level, down all layers until the qef
    // error is below threshold.
    // Returns the number of solved QEFs.
    pub fn solve_qefs(&self) -> usize {
        let mut num_solved = 0;
        if let Some(top_layer) = self.vertex_octtree.last() {
            for i in 0..top_layer.len() {
                num_solved += self.recursively_solve_qefs(&self.vertex_octtree.len() - 1, i);
            }
        }
        num_solved
    }

    fn recursively_solve_qefs(&self, layer: usize, index_in_layer: usize) -> usize {
        let vertex = &self.vertex_octtree[layer][index_in_layer];
        assert!(vertex.children.len() == 0 || layer > 0);
        let error;
        {
            // Solve qef and store error.
            let mut qef = vertex.qef.borrow_mut();
            // Make sure we never solve a qef twice.
            debug_assert!(
                qef.error.is_nan(),
                "found solved qef layer {:?} index {:?} {:?} parent: {:?}",
                layer,
                index_in_layer,
                vertex.index,
                vertex.parent
            );
            qef.solve();
            error = qef.error;
        }
        let mut num_solved = 1;
        // If error exceed threshold, recurse into subvertices.
        if Float::abs(error) > self.error {
            for &child_index in vertex.children.iter() {
                num_solved += self.recursively_solve_qefs(layer - 1, child_index);
            }
        }
        num_solved
    }

    // Generates leaf vertices along with a map that points VertexIndices to the index in the leaf
    // vertex vec.
    pub fn generate_leaf_vertices(&self) -> (Vec<Vertex<S>>, HashMap<VertexIndex, usize>) {
        let mut index_map = HashMap::new();
        let mut vertices = Vec::new();
        for edge_index in self.edge_grid.borrow().keys() {
            self.add_vertices_for_minimal_egde(edge_index, &mut vertices, &mut index_map);
        }
        for vertex in vertices.iter_mut() {
            for neighbor_vec in vertex.neighbors.iter_mut() {
                for neighbor in neighbor_vec.iter_mut() {
                    match neighbor {
                        &mut VarIndex::VertexIndex(vi) => {
                            *neighbor = VarIndex::Index(*index_map.get(&vi).unwrap())
                        }
                        &mut VarIndex::Index(_) => panic!("unexpected Index in fresh leaf map."),
                    }
                }
            }
        }
        for vi in 0..vertices.len() {
            for np in 0..vertices[vi].neighbors.len() {
                for ni in 0..vertices[vi].neighbors[np].len() {
                    match vertices[vi].neighbors[np][ni] {
                        VarIndex::VertexIndex(_) => panic!("unexpected VertexIndex."),
                        VarIndex::Index(i) => {
                            debug_assert!(
                                vertices[i].neighbors[np ^ 1].contains(&VarIndex::Index(vi)),
                                "vertex[{}].neighbors[{}][{}]=={:?},
                                 but vertex[{}].neighbors[{}]=={:?}\n{:?} vs. {:?}",
                                vi,
                                np,
                                ni,
                                vertices[vi].neighbors[np][ni],
                                i,
                                np ^ 1,
                                vertices[i].neighbors[np ^ 1],
                                vertices[vi],
                                vertices[i]
                            );
                        }
                    }
                }
            }
        }
        (vertices, index_map)
    }
    fn add_vertices_for_minimal_egde(
        &self,
        edge_index: &EdgeIndex,
        vertices: &mut Vec<Vertex<S>>,
        index_map: &mut HashMap<VertexIndex, usize>,
    ) {
        debug_assert!((edge_index.edge as usize) < 4);
        let cell_size = na::Vector3::new(self.res, self.res, self.res);
        for &quad_egde in QUADS[edge_index.edge as usize].iter() {
            let idx = neg_offset(edge_index.index, EDGE_OFFSET[quad_egde as usize]);

            let edge_set = get_connected_edges(quad_egde, self.bitset_for_cell(idx));
            let vertex_index = VertexIndex {
                edges: edge_set,
                index: idx,
            };
            index_map.entry(vertex_index).or_insert_with(|| {
                let mut neighbors = [
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ];
                for i in 0..6 {
                    if let Some(mut neighbor_index) = vertex_index.neighbor(i) {
                        for edges in get_connected_edges_from_edge_set(
                            neighbor_index.edges,
                            self.bitset_for_cell(neighbor_index.index),
                        ) {
                            neighbor_index.edges = edges;
                            let idx = VarIndex::VertexIndex(neighbor_index);
                            if !neighbors[i].contains(&idx) {
                                neighbors[i].push(idx);
                            }
                        }
                    }
                }
                let mut intersections = [0u32; 12];
                for edge in edge_set {
                    intersections[edge] = 1;
                }
                let tangent_planes: Vec<_> = edge_set
                    .into_iter()
                    .map(|edge| {
                        self.get_edge_tangent_plane(&EdgeIndex {
                            edge: Edge::from_usize(edge),
                            index: idx,
                        })
                    })
                    .collect();
                let cell_origin = self.origin
                    + na::Vector3::new(
                        From::from(idx[0] as f32),
                        From::from(idx[1] as f32),
                        From::from(idx[2] as f32),
                    ) * self.res;
                vertices.push(Vertex {
                    index: idx,
                    qef: RefCell::new(qef::Qef::new(
                        &tangent_planes,
                        BoundingBox::new(&cell_origin, &(cell_origin + cell_size)),
                    )),
                    neighbors: neighbors,
                    parent: Cell::new(None),
                    children: Vec::new(),
                    mesh_index: Cell::new(None),
                    edge_intersections: intersections,
                    euler_characteristic: 1,
                });
                vertices.len() - 1
            });
        }
    }

    fn get_edge_tangent_plane(&self, edge_index: &EdgeIndex) -> Plane<S> {
        if let Some(ref plane) = self.edge_grid.borrow().get(&edge_index.base()) {
            return *plane.clone();
        }
        panic!(
            "could not find edge_point: {:?} -> {:?}",
            edge_index,
            edge_index.base()
        );
    }

    // Return the Point index (in self.mesh.vertices) the the point belonging to edge/idx.
    fn lookup_cell_point(&self, edge: Edge, idx: Index) -> usize {
        // Generate the proper vertex Index from a single edge and an Index.
        let edge_set = get_connected_edges(edge, self.bitset_for_cell(idx));
        let vertex_index = VertexIndex {
            edges: edge_set,
            index: idx,
        };

        // Convert the vertex index to index and layer in the Octtree.
        let mut octtree_index = *self.vertex_index_map.get(&vertex_index).unwrap();
        let mut octtree_layer = 0;
        // Walk up the chain of parents
        loop {
            let next_index = self.vertex_octtree[octtree_layer][octtree_index]
                .parent
                .get()
                .unwrap();
            let ref next_vertex = self.vertex_octtree[octtree_layer + 1][next_index];
            let error = next_vertex.qef.borrow().error;
            if (!error.is_nan() && error > (self.error))
                || (octtree_layer == self.vertex_octtree.len() - 2)
                || !next_vertex.is_2manifold()
            {
                // Stop, if either the error is too large or we will reach the top.
                break;
            }
            octtree_layer += 1;
            octtree_index = next_index;
        }
        let vertex = &self.vertex_octtree[octtree_layer][octtree_index];
        // If the vertex exists in mesh, return its index.
        if let Some(mesh_index) = vertex.mesh_index.get() {
            return mesh_index;
        }
        // If not, store it in mesh and return its index.
        if vertex.qef.borrow().error.is_nan() {
            // Maybe the qef was not solved, since the error in the layer above was below the
            // threshold. But it seems, manifold criterion has catched and we need to solve it now.
            vertex.qef.borrow_mut().solve()
        }
        let qef_solution = vertex.qef.borrow().solution;
        let ref mut vertex_list = self.mesh.borrow_mut().vertices;
        let result = vertex_list.len();
        vertex.mesh_index.set(Some(result));
        vertex_list.push([qef_solution.x, qef_solution.y, qef_solution.z]);
        return result;
    }

    fn bitset_for_cell(&self, idx: Index) -> BitSet {
        let mut idx = idx;
        let mut result = BitSet::zero();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    if let Some(&v) = self.value_grid.get(&idx) {
                        if v < From::from(0f32) {
                            result.set(z << 2 | y << 1 | x);
                        }
                    } else {
                        panic!("did not find value_grid[{:?}]", idx);
                    }
                    idx[0] += 1;
                }
                idx[0] -= 2;
                idx[1] += 1;
            }
            idx[1] -= 2;
            idx[2] += 1;
        }
        result
    }

    // Compute a quad for the given edge and append it to the list.
    pub fn compute_quad(&self, edge_index: EdgeIndex) {
        debug_assert!((edge_index.edge as usize) < 4);
        debug_assert!(edge_index.index.iter().all(|&i| i > 0));

        let mut p = Vec::with_capacity(4);
        for &quad_egde in QUADS[edge_index.edge as usize].iter() {
            let point_index = self.lookup_cell_point(
                quad_egde,
                neg_offset(edge_index.index, EDGE_OFFSET[quad_egde as usize]),
            );
            // Dedup points before insertion (two minimal vertices might end up in the same parent
            // vertex).
            if !p.contains(&point_index) {
                p.push(point_index)
            }
        }
        // Only try to generate meshes, if there are more then two points.
        if p.len() < 3 {
            return;
        }
        // Reverse order, if the edge is reversed.
        if let Some(&v) = self.value_grid.get(&edge_index.index) {
            if v < From::from(0f32) {
                p.reverse();
            }
        }
        let ref mut face_list = self.mesh.borrow_mut().faces;
        // TODO: Fix this to choose the proper split.
        face_list.push([p[0], p[1], p[2]]);
        if p.len() == 4 {
            face_list.push([p[2], p[3], p[0]]);
        }
    }

    // If a is inside the object and b outside - this method returns the point on the line between
    // a and b where the object edge is. It also returns the normal on that point.
    // av and bv represent the object values at a and b.
    fn find_zero(&self, a: na::Point3<S>, av: S, b: na::Point3<S>, bv: S) -> Option<(Plane<S>)> {
        assert!(a != b);
        if Float::signum(av) == Float::signum(bv) {
            return None;
        }
        let d = a - b;
        let mut distance = Float::max(
            Float::max(Float::abs(d.x), Float::abs(d.y)),
            Float::abs(d.z),
        );
        distance = Float::min(Float::min(distance, Float::abs(av)), Float::abs(bv));
        let precision: S = From::from(PRECISION);
        if distance < precision * self.res {
            let mut result = &a;
            if Float::abs(bv) < Float::abs(av) {
                result = &b;
            }
            return Some(Plane {
                p: *result,
                // We need a precise normal here.
                n: self.function.normal(*result),
            });
        }
        // Linear interpolation of the zero crossing.
        let n = a + (b - a) * (Float::abs(av) / Float::abs(bv - av));
        let nv = self.function.value(n);

        if Float::signum(av) != Float::signum(nv) {
            return self.find_zero(a, av, n, nv);
        } else {
            return self.find_zero(n, nv, b, bv);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::get_connected_edges_from_edge_set;
    use super::super::bitset::BitSet;
    //  Corner indexes
    //
    //      6---------------7
    //     /|              /|
    //    / |             / |
    //   /  |            /  |
    //  4---------------5   |
    //  |   |           |   |
    //  |   2-----------|---3
    //  |  /            |  /
    //  | /             | /
    //  |/              |/
    //  0---------------1

    //  Edge indexes
    //
    //      +-------9-------+
    //     /|              /|
    //    7 |            10 |              ^
    //   /  8            /  11            /
    //  +-------6-------+   |     ^    higher indexes in y
    //  |   |           |   |     |     /
    //  |   +-------3---|---+     |    /
    //  2  /            5  /  higher indexes
    //  | 1             | 4      in z
    //  |/              |/        |/
    //  o-------0-------+         +-- higher indexes in x ---->

    #[test]
    fn connected_edges() {
        let cell = BitSet::from_4bits(0, 6, 3, 5);
        let edge_set = BitSet::from_4bits(4, 5, 10, 11);
        let connected_edges = get_connected_edges_from_edge_set(edge_set, cell);
        assert_eq!(connected_edges.len(), 2);
        assert!(connected_edges.contains(&BitSet::from_4bits(5, 5, 6, 10)));
        assert!(connected_edges.contains(&BitSet::from_4bits(3, 3, 4, 11)));
    }
}