// Copyright 2002 - 2008, 2010, 2011 National Technology Engineering
// Solutions of Sandia, LLC (NTESS). Under the terms of Contract
// DE-NA0003525 with NTESS, the U.S. Government retains certain rights
// in this software.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <Akri_AnalyticSurfaceInterfaceGeometry.hpp>
#include <Akri_CDFEM_Parent_Edges.hpp>
#include <Akri_CDFEM_Support.hpp>
#include <Akri_MathUtil.hpp>
#include <Akri_MeshHelpers.hpp>
#include <Akri_Surface.hpp>
#include <Akri_MasterElementDeterminer.hpp>
#include "Akri_DiagWriter.hpp"
#include "Akri_PhaseTag.hpp"
#include "Akri_SnapInfo.hpp"

namespace krino {

static int surface_sign_at_position(const Surface & surface, const Vector3d & pt)
{
  const double phi = surface.point_signed_distance(pt);
  return ( (phi < 0.) ? -1 : 1 ); // GOMA sign convention
}

static std::function<double(const double)> build_edge_distance_function(const Surface & surface, const Segment3d & edge)
{
  std::function<double(const double)> distanceFunction =
    [&surface, &edge](const double x)
    {
      return surface.point_signed_distance((1.-x)*edge.GetNode(0) + x*edge.GetNode(1));
    };
  return distanceFunction;
}

static double find_crossing_position(const Surface & surface, const Segment3d & edge)
{
  const double phi0 = surface.point_signed_distance(edge.GetNode(0));
  const double phi1 = surface.point_signed_distance(edge.GetNode(1));
  const auto result = find_root(build_edge_distance_function(surface, edge), 0., 1., phi0, phi1);
  ThrowRequire(result.first);
  return result.second;
}

static int compute_element_sign(const Surface & surface, const std::vector<Vector3d> & elemNodesCoords)
{
  int crossingState = 0;
  for(auto && nodeCoords : elemNodesCoords)
  {
    crossingState = crossingState | ((surface.point_signed_distance(nodeCoords) < 0.) ? 1 : 2);
    if (crossingState == 3) return 0;
  }
  ThrowAssert(crossingState == 1 || crossingState == 2);
  return (crossingState == 1) ? -1 : 1;
}

static Vector3d get_centroid(const std::vector<Vector3d> & elemNodesCoords)
{
  Vector3d centroid = Vector3d::ZERO;
  for(auto && nodeCoords : elemNodesCoords)
  {
    centroid += nodeCoords;
  }
  centroid *= 1./elemNodesCoords.size();
  return centroid;
}

SurfaceElementCutter::SurfaceElementCutter(const stk::mesh::BulkData & mesh,
  stk::mesh::Entity element,
  const Surface & surface)
: myMasterElem(MasterElementDeterminer::getMasterElement(mesh.bucket(element).topology())),
  mySurface(surface)
{
  const FieldRef coordsField(mesh.mesh_meta_data().coordinate_field());
  fill_element_node_coordinates(mesh, element, coordsField, myElementNodeCoords);
  myElementSign = compute_element_sign(surface, myElementNodeCoords);
}

std::vector<InterfaceID> SurfaceElementCutter::get_sorted_cutting_interfaces() const
{
  std::vector<InterfaceID> interfaces;
  if (0 == myElementSign)
    interfaces.push_back(InterfaceID(0,0));
  return interfaces;
}

bool SurfaceElementCutter::have_crossing(const InterfaceID interface, const Segment3d & edge) const
{
  return surface_sign_at_position(mySurface, parametric_to_global_coordinates(edge.GetNode(0))) !=
         surface_sign_at_position(mySurface, parametric_to_global_coordinates(edge.GetNode(1)));
}

double SurfaceElementCutter::interface_crossing_position(const InterfaceID interface, const Segment3d & edge) const
{
  const Segment3d globalEdge(parametric_to_global_coordinates(edge.GetNode(0)), parametric_to_global_coordinates(edge.GetNode(1)));
  return find_crossing_position(mySurface, globalEdge);
}

int SurfaceElementCutter::sign_at_position(const InterfaceID interface, const Vector3d & paramCoords) const
{
  return surface_sign_at_position(mySurface, parametric_to_global_coordinates(paramCoords));
}

Vector3d SurfaceElementCutter::parametric_to_global_coordinates(const Vector3d & pCoords) const
{
  std::vector<double> nodalShapeFunctions(myMasterElem.num_nodes());
  myMasterElem.shape_fcn(1, pCoords.data(), nodalShapeFunctions.data());
  Vector3d pt(Vector3d::ZERO);
  for (unsigned n=0; n<myMasterElem.num_nodes(); ++n)
    pt += nodalShapeFunctions[n] * myElementNodeCoords[n];
  return pt;
}

static void append_surface_edge_intersection_points(const stk::mesh::BulkData & mesh,
    const std::vector<stk::mesh::Entity> & elementsToIntersect,
    const Surface & surface,
    const IntersectionPointFilter & intersectionPointFilter,
    std::vector<IntersectionPoint> & intersectionPoints)
{
  const bool intersectionPointIsOwned = true;
  std::vector<int> intersectionPointSortedDomains;
  const int dim = mesh.mesh_meta_data().spatial_dimension();
  const FieldRef coordsField(mesh.mesh_meta_data().coordinate_field());
  std::set<std::array<stk::mesh::EntityId,2>> edgesAlreadyChecked;
  for (auto && elem : elementsToIntersect)
  {
    const stk::topology topology = mesh.bucket(elem).topology();
    const stk::mesh::Entity* elem_nodes = mesh.begin_nodes(elem);
    const unsigned numEdges = topology.num_edges();

    for (unsigned iedge = 0; iedge < numEdges; ++iedge)
    {
      const unsigned * edge_node_ordinals = get_edge_node_ordinals(topology, iedge);
      const stk::mesh::Entity node0 = elem_nodes[edge_node_ordinals[0]];
      const stk::mesh::Entity node1 = elem_nodes[edge_node_ordinals[1]];
      const stk::mesh::EntityId node0Id = mesh.identifier(node0);
      const stk::mesh::EntityId node1Id = mesh.identifier(node1);
      const std::array<stk::mesh::EntityId,2> edgeNodeIds = (node0Id < node1Id) ? std::array<stk::mesh::EntityId,2>{node0Id, node1Id} : std::array<stk::mesh::EntityId,2>{node1Id, node0Id};
      auto iter = edgesAlreadyChecked.lower_bound(edgeNodeIds);
      if (iter == edgesAlreadyChecked.end() || edgeNodeIds != *iter)
      {
        edgesAlreadyChecked.insert(iter, edgeNodeIds);
        const Vector3d node0Coords(field_data<double>(coordsField, node0), dim);
        const Vector3d node1Coords(field_data<double>(coordsField, node1), dim);
        const double phi0 = surface.point_signed_distance(node0Coords);
        const double phi1 = surface.point_signed_distance(node1Coords);
        const bool haveCrossing = (phi0 < 0.) ? (phi1 >= 0.) : (phi1 < 0.);
        if (haveCrossing)
        {
          const InterfaceID interface(0,0);
          const double location = find_crossing_position(surface, Segment3d(node0Coords, node1Coords));
          interface.fill_sorted_domains(intersectionPointSortedDomains);
          const std::vector<stk::mesh::Entity> intersectionPointNodes{node0,node1};
          if (intersectionPointFilter(intersectionPointNodes, intersectionPointSortedDomains))
            intersectionPoints.emplace_back(intersectionPointIsOwned, intersectionPointNodes, std::vector<double>{1.-location, location}, intersectionPointSortedDomains);
        }
      }
    }
  }
}

static BoundingBox compute_nodal_bounding_box(const stk::mesh::BulkData & mesh)
{
  const int nDim = mesh.mesh_meta_data().spatial_dimension();
  const FieldRef coordsField(mesh.mesh_meta_data().coordinate_field());

  BoundingBox nodeBbox;
  for ( auto && bucket : mesh.buckets(stk::topology::NODE_RANK) )
  {
    double *coord = field_data<double>(coordsField, *bucket);
    for (size_t n = 0; n < bucket->size(); ++n)
      nodeBbox.accommodate( Vector3d(coord+n*nDim, nDim) );
  }

  return nodeBbox;
}

static void prepare_to_compute_with_surface(const stk::mesh::BulkData & mesh, const Surface & surface)
{
  const BoundingBox nodeBbox = compute_nodal_bounding_box(mesh);
  Surface & nonConstSurface = const_cast<Surface&>(surface);
  nonConstSurface.prepare_to_compute(0.0, nodeBbox, 0.); // Setup including communication of facets that are within this processors narrow band
}

void AnalyticSurfaceInterfaceGeometry::prepare_to_process_elements(const stk::mesh::BulkData & mesh,
    const NodeToCapturedDomainsMap & nodesToCapturedDomains) const
{
  myElementsToIntersect = get_owned_parent_elements(mesh, myActivePart, myCdfemSupport, myPhaseSupport);
  prepare_to_compute_with_surface(mesh, mySurface);
}

void AnalyticSurfaceInterfaceGeometry::prepare_to_process_elements(const stk::mesh::BulkData & mesh,
  const std::vector<stk::mesh::Entity> & elementsToIntersect,
  const NodeToCapturedDomainsMap & nodesToCapturedDomains) const
{
  myElementsToIntersect = elementsToIntersect;
  prepare_to_compute_with_surface(mesh, mySurface);
}

static bool edge_is_possibly_cut(const std::array<Vector3d,2> & edgeNodeCoords, const std::array<double,2> & edgeNodeDist)
{
  const double edgeLength = (edgeNodeCoords[1]-edgeNodeCoords[0]).length();

  return std::abs(edgeNodeDist[0]) < edgeLength && std::abs(edgeNodeDist[1]) < edgeLength;
}

static bool element_has_possibly_cut_edge(stk::topology elemTopology, const std::vector<Vector3d> & elemNodeCoords, const std::vector<double> & elemNodeDist)
{
  const unsigned numEdges = elemTopology.num_edges();
  for(unsigned i=0; i < numEdges; ++i)
  {
    const unsigned * edgeNodeOrdinals = get_edge_node_ordinals(elemTopology, i);
    if (edge_is_possibly_cut({{elemNodeCoords[edgeNodeOrdinals[0]], elemNodeCoords[edgeNodeOrdinals[1]]}}, {{elemNodeDist[edgeNodeOrdinals[0]], elemNodeDist[edgeNodeOrdinals[1]]}}))
      return true;
  }
  return false;
}

static void fill_point_distances(const Surface & surface, const std::vector<Vector3d> & points, std::vector<double> & pointDist)
{
  pointDist.clear();
  for (auto && point : points)
    pointDist.push_back(surface.point_signed_distance(point));
}

std::vector<stk::mesh::Entity> AnalyticSurfaceInterfaceGeometry::get_possibly_cut_elements(const stk::mesh::BulkData & mesh) const
{
  NodeToCapturedDomainsMap nodesToSnappedDomains;
  prepare_to_process_elements(mesh, nodesToSnappedDomains);

  std::vector<stk::mesh::Entity> possibleCutElements;
  std::vector<Vector3d> elementNodeCoords;
  std::vector<double> elementNodeDist;
  const FieldRef coordsField(mesh.mesh_meta_data().coordinate_field());

  const stk::mesh::Selector activeLocallyOwned = myActivePart & mesh.mesh_meta_data().locally_owned_part();

  for(const auto & bucketPtr : mesh.get_buckets(stk::topology::ELEMENT_RANK, activeLocallyOwned))
  {
    for(const auto & elem : *bucketPtr)
    {
      fill_element_node_coordinates(mesh, elem, coordsField, elementNodeCoords);
      fill_point_distances(mySurface, elementNodeCoords, elementNodeDist);
      if (element_has_possibly_cut_edge(bucketPtr->topology(), elementNodeCoords, elementNodeDist))
        possibleCutElements.push_back(elem);
    }
  }

  return possibleCutElements;
}

static bool all_nodes_of_element_will_be_snapped(const stk::mesh::BulkData & mesh,
    stk::mesh::Entity element,
    stk::mesh::Entity snapNode,
    const NodeToCapturedDomainsMap & nodesToCapturedDomains)
{
  for (auto node : StkMeshEntities{mesh.begin_nodes(element), mesh.end_nodes(element)})
    if (node != snapNode && nodesToCapturedDomains.find(node) == nodesToCapturedDomains.end())
      return false;
  return true;
}

static void set_domains_for_element_if_it_will_be_uncut_after_snapping(const stk::mesh::BulkData & mesh,
    const Surface & surface,
    stk::mesh::Entity element,
    stk::mesh::Entity snapNode,
    const NodeToCapturedDomainsMap & nodesToCapturedDomains,
    ElementToDomainMap & elementsToDomain )
{
  auto iter = elementsToDomain.lower_bound(element);
  if (iter == elementsToDomain.end() || iter->first != element)
  {
    if (all_nodes_of_element_will_be_snapped(mesh, element, snapNode, nodesToCapturedDomains))
    {
      const FieldRef coordsField(mesh.mesh_meta_data().coordinate_field());
      std::vector<Vector3d> elemNodesCoords;
      fill_element_node_coordinates(mesh, element, coordsField, elemNodesCoords);
      const int elementSign = surface_sign_at_position(surface, get_centroid(elemNodesCoords));

      elementsToDomain.emplace_hint(iter, element, elementSign);
    }
  }
}

void AnalyticSurfaceInterfaceGeometry::store_phase_for_elements_that_will_be_uncut_after_snapping(const stk::mesh::BulkData & mesh,
      const std::vector<IntersectionPoint> & intersectionPoints,
      const std::vector<SnapInfo> & snapInfos,
      const NodeToCapturedDomainsMap & nodesToCapturedDomains) const
{
  for (auto && snapInfo : snapInfos)
  {
    stk::mesh::Entity snapNode = mesh.get_entity(stk::topology::NODE_RANK, snapInfo.get_node_global_id());
    for (auto elem : StkMeshEntities{mesh.begin_elements(snapNode), mesh.end_elements(snapNode)})
      if (mesh.bucket(elem).owned() && mesh.bucket(elem).member(myActivePart))
        set_domains_for_element_if_it_will_be_uncut_after_snapping(mesh, mySurface, elem, snapNode, nodesToCapturedDomains, myUncutElementPhases);
  }
}

std::vector<IntersectionPoint> AnalyticSurfaceInterfaceGeometry::get_edge_intersection_points(const stk::mesh::BulkData & mesh,
    const NodeToCapturedDomainsMap & nodesToCapturedDomains) const
{
  NodeToCapturedDomainsMap nodesToSnappedDomains;
  prepare_to_process_elements(mesh, nodesToSnappedDomains);

  const IntersectionPointFilter intersectionPointFilter = keep_all_intersection_points_filter();
  std::vector<IntersectionPoint> intersectionPoints;
  append_surface_edge_intersection_points(mesh, myElementsToIntersect, mySurface, intersectionPointFilter, intersectionPoints);
  return intersectionPoints;
}

void AnalyticSurfaceInterfaceGeometry::append_element_intersection_points(const stk::mesh::BulkData & mesh,
  const NodeToCapturedDomainsMap & nodesToCapturedDomains,
  const std::vector<stk::mesh::Entity> & elementsToIntersect,
  const IntersectionPointFilter & intersectionPointFilter,
  std::vector<IntersectionPoint> & intersectionPoints) const
{
  prepare_to_process_elements(mesh, elementsToIntersect, nodesToCapturedDomains);
  append_surface_edge_intersection_points(mesh, myElementsToIntersect, mySurface, intersectionPointFilter, intersectionPoints);
}

std::unique_ptr<ElementCutter> AnalyticSurfaceInterfaceGeometry::build_element_cutter(const stk::mesh::BulkData & mesh,
  stk::mesh::Entity element,
  const std::function<bool(const std::array<unsigned,4> &)> & intersectingPlanesDiagonalPicker) const
{
  std::unique_ptr<ElementCutter> cutter;
  cutter.reset( new SurfaceElementCutter(mesh, element, mySurface) );
  return cutter;
}

PhaseTag AnalyticSurfaceInterfaceGeometry::get_starting_phase(const ElementCutter * cutter) const
{
  const SurfaceElementCutter * surfaceCutter = dynamic_cast<const SurfaceElementCutter *>(cutter);
  ThrowRequire(surfaceCutter);

  PhaseTag phase;
  ThrowRequire(1 == mySurfaceIdentifiers.size());
  phase.add(mySurfaceIdentifiers[0], surfaceCutter->get_element_sign());
  return phase;
}

} // namespace krino
