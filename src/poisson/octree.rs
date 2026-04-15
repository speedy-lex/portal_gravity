use core::slice;
use std::iter::{self, FlatMap};

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct Octree {
    pos: Vec3,
    size: f32,
    children: OctreeInner,
}

#[derive(Debug, Clone)]
enum OctreeInner {
    Points(Vec<Vec3>),
    Octree(Box<[Octree; 8]>),
}

impl Octree {
    pub fn new(pos: Vec3, size: f32, divisions: usize) -> Self {
        if divisions == 0 {
            return Self {
                pos,
                size,
                children: OctreeInner::Points(vec![]),
            };
        }

        let children = OctreeInner::Octree(Box::new(std::array::from_fn(|x| {
            let offset = Vec3::new(
                (x & 0b1) as f32,
                ((x & 0b10) >> 1) as f32,
                ((x & 0b100) >> 2) as f32,
            ) * size
                / 2.0;
            Self::new(pos + offset, size / 2.0, divisions - 1)
        })));

        Self {
            pos,
            size,
            children,
        }
    }

    pub fn get_node(&self, x: usize, y: usize, z: usize, depth: usize) -> &Octree {
        if depth == 0 {
            return self;
        }

        match &self.children {
            OctreeInner::Points(_) => self,
            OctreeInner::Octree(children) => {
                let mask = 1 << (depth - 1);
                let index = ((x & mask) >> (depth - 1))
                    | ((y & mask) >> (depth - 1) << 1)
                    | ((z & mask) >> (depth - 1) << 2);
                children[index].get_node(x, y, z, depth - 1)
            }
        }
    }

    pub fn insert(&mut self, point: Vec3) {
        match &mut self.children {
            OctreeInner::Points(points) => points.push(point),
            OctreeInner::Octree(children) => {
                let center = self.pos + (self.size / 2.0);
                let x = point.x > center.x;
                let y = point.y > center.y;
                let z = point.z > center.z;

                let index = ((z as usize) << 2) | ((y as usize) << 1) | (x as usize);

                children[index].insert(point);
            }
        }
    }

    pub fn iter(&self) -> Iter<'_> {
        match &self.children {
            OctreeInner::Points(points) => Iter::Points(points.iter().copied()),
            OctreeInner::Octree(children) => {
                Iter::Octree(children.iter().flat_map(Octree::boxed_iter))
            }
        }
    }
    fn boxed_iter(&self) -> Box<Iter<'_>> {
        Box::new(self.iter())
    }

    pub fn pos(&self) -> Vec3 {
        self.pos
    }
    pub fn size(&self) -> f32 {
        self.size
    }
}

#[allow(clippy::type_complexity)]
pub enum Iter<'a> {
    Points(iter::Copied<slice::Iter<'a, Vec3>>),
    Octree(FlatMap<slice::Iter<'a, Octree>, Box<Iter<'a>>, fn(&'a Octree) -> Box<Iter<'a>>>),
}
impl<'a> Iterator for Iter<'a> {
    type Item = Vec3;

    fn next(&mut self) -> Option<Self::Item> {
        use Iter::*;

        match self {
            Points(x) => x.next(),
            Octree(x) => x.next(),
        }
    }
}
