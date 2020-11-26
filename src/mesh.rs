use alga::general::RealField;
use nalgebra as na;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::fmt::Debug;

#[cfg(feature = "obj")]
use std::{error::Error, fs::File, io::Write, path::Path};

#[cfg(feature = "polyhedron-ops")]
use polyhedron_ops as p_ops;

/// A polygon mesh consiting of (mostly) quads and triangles.
///
/// This can be tessellated further into a pure [`TriangleMesh`].
///
/// Returned from
/// [`ManifoldDualContouring::tessellate()`](crate::ManifoldDualContouring::tessellate()).
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh<S: Clone> {
    /// The list of vertices.
    pub vertices: Vec<[S; 3]>,
    /// The list of faces as indexes into vertices.
    pub faces: Vec<SmallVec<[usize; 4]>>,
}

impl<S: Clone> Mesh<S> {
    /// Tessellates the mesh into triangles and yields a
    /// [`TriangleMesh`].
    pub fn to_triangle_mesh(&self) -> TriangleMesh<S> {
        TriangleMesh {
            vertices: self.vertices.clone(),
            faces: self
                .faces
                .par_iter()
                .flat_map(|face| {
                    if 4 == face.len() {
                        vec![[face[0], face[1], face[2]], [face[2], face[3], face[0]]]
                    } else {
                        vec![[face[0], face[1], face[2]]]
                    }
                })
                .collect::<Vec<[usize; 3]>>(),
        }
    }

    /// Returns the mesh’s topology as two, flat buffers.
    ///
    /// The first buffer contains the number of vertices per face (also
    /// called the *arity of the face*).
    ///
    /// The 2nd buffer contains the index into the `vertices` array of
    /// the underlying [`Mesh`].
    pub fn flat_topology(&self) -> (Vec<usize>, Vec<usize>) {
        let mut face_arities = Vec::with_capacity(self.faces.len());
        let faces = self
            .faces
            .iter()
            .flat_map(|face| {
                face_arities.push(face.len());
                face.clone()
            })
            .collect();
        (face_arities, faces)
    }

    /// Describe the mesh as an [`Vec<u8>`] buffer containing a
    /// [Wavefront OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
    /// file.
    ///
    /// Depending on the target coordinate system (left- or right
    /// handed) the mesh’s winding order can be reversed with the
    /// `reverse_face_winding` flag.
    #[cfg(feature = "obj")]
    pub fn to_obj(
        &self,
        reverse_face_winding: bool,
    ) -> Result<Vec<u8>, Box<dyn Error>>
    where
        S: Into<f32>,
    {
        let mut file = Vec::new();

        writeln!(file, "o SDFMesh")?;

        for vertex in &self.vertices {
            writeln!(
                file,
                "v {} {} {}",
                vertex[0].clone().into(),
                vertex[1].clone().into(),
                vertex[2].clone().into()
            )?;
        }

        match reverse_face_winding {
            true => {
                for face in &self.faces {
                    write!(file, "f")?;
                    for vertex_index in face.iter().rev() {
                        write!(file, " {}", vertex_index + 1)?;
                    }
                    writeln!(file, "")?;
                }
            }
            false => {
                for face in &self.faces {
                    write!(file, "f")?;
                    for vertex_index in face {
                        write!(file, " {}", vertex_index + 1)?;
                    }
                    writeln!(file, "")?;
                }
            }
        };

        Ok(file)
    }

    /// Export the mesh as a
    /// [Wavefront OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
    /// file.
    ///
    /// Depending on the target coordinate system (left- or right
    /// handed) the mesh’s winding order can be reversed with the
    /// `reverse_face_winding` flag.
    #[cfg(feature = "obj")]
    pub fn export_as_obj(
        &self,
        destination: &Path,
        reverse_face_winding: bool,
    ) -> Result<(), Box<dyn Error>>
    where S: Into<f32>
    {
        let mut file = File::create(destination)?;
        file.write_all(&self.to_obj(reverse_face_winding)?)?;
        file.flush()?;

        Ok(())
    }
}

#[cfg(feature = "polyhedron-ops")]
impl<S: Clone + Into<f32>> From<Mesh<S>> for p_ops::Polyhedron {
    fn from(mesh: Mesh<S>) -> p_ops::Polyhedron
    where
        S: Into<f32>,
    {
        p_ops::Polyhedron::from(
            "SDFMesh",
            mesh
                .vertices
                .iter()
                .map(|vertex| {
                    p_ops::Point::new(
                        vertex[0].clone().into(),
                        vertex[1].clone().into(),
                        vertex[2].clone().into(),
                    )
                })
                .collect(),
            mesh
                .faces
                .iter()
                .map(|face| face.iter().map(|index| *index as u32).collect())
                .collect(),
            None,
        )
    }
}

/// Triangle mesh that will be returned from
/// [`TriangleMesh::from<Mesh>()`] or
/// [`Mesh::to_triangle_mesh()`].
#[derive(Clone, Debug, PartialEq)]
pub struct TriangleMesh<S: Clone> {
    /// The list of vertices.
    pub vertices: Vec<[S; 3]>,
    /// The list of triangles as indexes into vertices.
    pub faces: Vec<[usize; 3]>,
}

/// Converts
impl<S: Clone> From<Mesh<S>> for TriangleMesh<S> {
    fn from(mesh: Mesh<S>) -> Self {
        mesh.to_triangle_mesh()
    }
}

impl<S: Clone> TriangleMesh<S> {
    /// Returns the mesh’s topology as a flat buffer.
    ///
    /// Each triangle is represented by a group of three entries into
    /// the returned buffer.
    pub fn flat_topology(&self) -> Vec<usize> {
        self.faces
            .par_iter()
            .flat_map(|face| face.to_vec())
            .collect()
    }
}

impl<S: RealField + Debug> TriangleMesh<S> {
    /// Return the normal of the face at index `face` as triple of
    /// `f32`s.
    pub fn normal<T>(&self, face: usize) -> [T; 3]
    where
        f32: From<S>,
        T: From<f32>,
    {
        let v: Vec<na::Point3<f32>> = self.faces[face]
            .par_iter()
            .map(|&i| {
                na::Point3::<f32>::new(
                    self.vertices[i][0].into(),
                    self.vertices[i][1].into(),
                    self.vertices[i][2].into(),
                )
            })
            .collect();
        let r = (v[1] - v[0]).cross(&(v[2] - v[0])).normalize();
        [r[0].into(), r[1].into(), r[2].into()]
    }

    /// Return the vertics of the face at index `i` as triple of
    /// `f32`s.
    pub fn vertex<T>(&self, i: usize) -> [T; 3]
    where
        T: From<S>,
    {
        [
            self.vertices[i][0].into(),
            self.vertices[i][1].into(),
            self.vertices[i][2].into(),
        ]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn f32slice_eq(a: &[f32], b: &[f32]) -> bool {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if (a[i] - b[i]).abs() > f32::EPSILON {
                return false;
            }
        }
        true
    }

    #[test]
    fn simple() {
        let m = TriangleMesh {
            vertices: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
            faces: vec![[0, 1, 2]],
        };
        assert!(f32slice_eq(&m.normal::<f32>(0), &[0., 0., 1.]));
        assert!(f32slice_eq(&m.vertex::<f32>(0), &[0., 0., 0.]));
        assert!(f32slice_eq(&m.vertex::<f32>(1), &[1., 0., 0.]));
        assert!(f32slice_eq(&m.vertex::<f32>(2), &[0., 1., 0.]));
    }
}
