use nalgebra as na;
use std::path::PathBuf;
struct UnitSphere {
    bbox: tessellation::BoundingBox<f64>,
}

impl UnitSphere {
    fn new() -> UnitSphere {
        UnitSphere {
            bbox: tessellation::BoundingBox::new(
                &na::Point3::new(-1., -1., -1.),
                &na::Point3::new(1., 1., 1.),
            ),
        }
    }
}

impl tessellation::ImplicitFunction<f64> for UnitSphere {
    fn bbox(&self) -> &tessellation::BoundingBox<f64> {
        &self.bbox
    }
    fn value(&self, p: &na::Point3<f64>) -> f64 {
        return na::Vector3::new(p.x, p.y, p.z).norm() - 1.0;
    }
    fn normal(&self, p: &na::Point3<f64>) -> na::Vector3<f64> {
        return na::Vector3::new(p.x, p.y, p.z).normalize();
    }
}

fn main() {
    let sphere = UnitSphere::new();
    let mut mdc = tessellation::ManifoldDualContouring::new(&sphere, 0.2, 0.1);
    let mesh = mdc.tessellate().unwrap();

    mesh.write_to_obj(&PathBuf::from("foo.obj"), false);
}
