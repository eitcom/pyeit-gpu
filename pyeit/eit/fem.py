# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-instance-attributes
""" 2D/3D FEM routines """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function, annotations

from typing import Tuple, Union
import cupy as cp
import cupy.linalg as cla
import warnings
from cupyx.scipy import sparse
import cupyx.scipy.sparse.linalg
from cupyx.scipy.sparse import csr_matrix

from pyeit.eit.protocol import PyEITProtocol
from pyeit.mesh import PyEITMesh


class Forward:
    """FEM forward computing code"""

    def __init__(self, mesh: PyEITMesh) -> None:
        """
        FEM forward solver.
        A good FEM forward solver should only depend on
        mesh structure and the position of electrodes.

        Parameters
        ----------
        mesh: PyEITMesh
            mesh object

        Note
        ----
        The nodes are continuous numbered, the numbering of an element is
        CCW (counter-clock-wise).
        """
        self.mesh = mesh
        # coefficient matrix [initialize]
        self.se = calculate_ke(self.mesh.node, self.mesh.element)
        self.assemble_pde(self.mesh.perm)

    def assemble_pde(self, perm: Union[int, float, cp.ndarray]) -> None:
        """
        assemble PDE

        Parameters
        ----------
        perm : Union[int, float, cp.ndarray]
            permittivity on elements ; shape (n_tri,).
            if `None`, assemble_pde is aborded

        """
        if perm is None:
            return
        perm = self.mesh.get_valid_perm(perm)
        self.kg = assemble(
            self.se, self.mesh.element, perm, self.mesh.n_nodes, ref=self.mesh.ref_node
        )

    def solve(self, ex_line: cp.ndarray = None) -> cp.ndarray:
        """
        Calculate and compute the potential distribution (complex-valued)
        corresponding to the permittivity distribution `perm ` for a
        excitation contained specified by `ex_line` (Neumann BC)

        Parameters
        ----------
        ex_line : cp.ndarray, optional
            stimulation/excitation matrix, of shape (2,)

        Returns
        -------
        cp.ndarray
            potential on nodes ; shape (n_pts,)

        Notes
        -----
        Currently, only simple electrode model is supported,
        CEM (complete electrode model) is under development.
        """
        # using natural boundary conditions
        b = cp.zeros(self.mesh.n_nodes)
        b[self.mesh.el_pos[ex_line]] = [1, -1]

        # solve
        return sparse.linalg.spsolve(self.kg, b)


class EITForward(Forward):
    """EIT Forward simulation, depends on mesh and protocol"""

    def __init__(self, mesh: PyEITMesh, protocol: PyEITProtocol) -> None:
        """
        EIT Forward Solver

        Parameters
        ----------
        mesh: PyEITMesh
            mesh object
        protocol: PyEITProtocol
            measurement object

        Notes
        -----
        The Jacobian and the boundary voltages used the SIGN information,
        for example, V56 = V6 - V5 = -V65. If you are using absolute boundary
        voltages for imaging, you MUST normalize it with the signs of v0
        under each current-injecting pattern.
        """
        self._check_mesh_protocol_compatibility(mesh, protocol)

        # FEM solver
        super().__init__(mesh=mesh)

        # EIT measurement protocol
        self.protocol = protocol

    def _check_mesh_protocol_compatibility(
        self, mesh: PyEITMesh, protocol: PyEITProtocol
    ) -> None:
        """
        Check if mesh and protocol are compatible

        - #1 n_el in mesh >=  n_el in protocol
        - #2 .., TODO if necessary

        Raises
        ------
        ValueError
            if protocol is not compatible to the mesh
        """
        # n_el in mesh should be >=  n_el in protocol
        m_n_el = mesh.n_el
        p_n_el = protocol.n_el

        if m_n_el != p_n_el:
            warnings.warn(
                f"The mesh use {m_n_el} electrodes, and the protocol use only {p_n_el} electrodes",
                stacklevel=2,
            )

        if m_n_el < p_n_el:
            raise ValueError(
                f"Protocol is not compatible with mesh :\
The mesh use {m_n_el} electrodes, and the protocol use only {p_n_el} electrodes "
            )

    def solve_eit(
        self,
        perm: Union[int, float, cp.ndarray] = None,
    ) -> cp.ndarray:
        """
        EIT simulation, generate forward v measurements

        Parameters
        ----------
        perm : Union[int, float, cp.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of forward v measurements will be
            based on the permittivity of the mesh, self.mesh.perm
        Returns
        -------
        v: cp.ndarray
            simulated boundary voltage measurements; shape(n_exe*n_el,)
        """
        self.assemble_pde(perm)
        v = cp.zeros(
            (self.protocol.n_exc, self.protocol.n_meas), dtype=self.mesh.perm.dtype
        )
        for i, ex_line in enumerate(self.protocol.ex_mat):
            f = self.solve(ex_line)
            v[i] = subtract_row(f[self.mesh.el_pos], self.protocol.meas_mat[i])

        return cp.asarray(v.reshape(-1))

    def compute_jac(
        self,
        perm: Union[int, float, cp.ndarray] = None,
        normalize: bool = False,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Compute the Jacobian matrix and initial boundary voltage meas.
        extimation v0

        Parameters
        ----------
        perm : Union[int, float, cp.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of Jacobian matrix will be based
            on the permittivity of the mesh, self.mesh.perm
        normalize : bool, optional
            flag for Jacobian normalization, by default False.
            If True the Jacobian is normalized

        Returns
        -------
        Tuple[cp.ndarray, cp.ndarray]
            Jacobian matrix, initial boundary voltage meas. extimation v0

        """
        # update k if necessary and calculate r=inv(k)
        self.assemble_pde(perm)
        r_el = cla.inv(self.kg.toarray())[self.mesh.el_pos]

        # calculate v, jac per excitation pattern (ex_line)
        _jac = cp.zeros(
            (self.protocol.n_exc, self.protocol.n_meas, self.mesh.n_elems),
            dtype=self.mesh.perm.dtype,
        )
        v = cp.zeros(
            (self.protocol.n_exc, self.protocol.n_meas), dtype=self.mesh.perm.dtype
        )
        for i, ex_line in enumerate(self.protocol.ex_mat):
            f = self.solve(ex_line)
            v[i] = subtract_row(f[self.mesh.el_pos], self.protocol.meas_mat[i])
            ri = subtract_row(r_el, self.protocol.meas_mat[i])
            # Build Jacobian matrix column wise (element wise)
            #    Je = Re*Ke*Ve = (nex3) * (3x3) * (3x1)
            for (e, ijk) in enumerate(self.mesh.element.get()):
                tmp = cp.dot(ri[:, ijk], self.se[e])
                _jac[i, :, e] = cp.dot(tmp, f[ijk])
        # measurement protocol
        jac = cp.vstack(_jac)
        v0 = v.reshape(-1)

        # Jacobian normalization: divide each row of J (J[i]) by abs(v0[i])
        if normalize:
            jac = jac / cp.abs(v0[:, None])
        return jac, v0

    def compute_b_matrix(
        self,
        perm: Union[int, float, cp.ndarray] = None,
    ) -> cp.ndarray:
        """
        Compute back-projection mappings (smear matrix)

        Parameters
        ----------
        perm : Union[int, float, cp.ndarray], optional
            permittivity on elements ; shape (n_tri,), by default `None`.
            if perm is `None`, the computation of smear matrix will be based
            on the permittivity of the mesh, self.mesh.perm

        Returns
        -------
        cp.ndarray
            back-projection mappings (smear matrix); shape(n_exc, n_pts, 1), dtype= bool
        """
        self.assemble_pde(perm)
        b_mat = cp.zeros((self.protocol.n_exc, self.protocol.n_meas, self.mesh.n_nodes))

        for i, ex_line in enumerate(self.protocol.ex_mat):
            f = self.solve(ex_line=ex_line)
            f_el = f[self.mesh.el_pos]
            # build bp projection matrix
            # 1. we can either smear at the center of elements, using
            #    >> fe = cp.mean(f[:, self.tri], axis=1)
            # 2. or, simply smear at the nodes using f
            b_mat[i] = _smear(f, f_el, self.protocol.meas_mat[i])

        return cp.vstack(b_mat)


def _smear(f: cp.ndarray, fb: cp.ndarray, pairs: cp.ndarray) -> cp.ndarray:
    """
    Build smear matrix B for bp for one exitation

    used for the smear matrix computation by @ChabaneAmaury

    Parameters
    ----------
    f: cp.ndarray
        potential on nodes
    fb: cp.ndarray
        potential on adjacent electrodes
    pairs: cp.ndarray
        electrodes numbering pairs

    Returns
    -------
    B: cp.ndarray
        back-projection matrix
    """
    # Replacing the code below by a faster implementation in Numpy
    f_min = cp.minimum(fb[pairs[:, 0]], fb[pairs[:, 1]]).reshape((-1, 1))
    f_max = cp.maximum(fb[pairs[:, 0]], fb[pairs[:, 1]]).reshape((-1, 1))
    return (f_min < f) & (f <= f_max)


def smear_nd(
    f: cp.ndarray, fb: cp.ndarray, meas_pattern: cp.ndarray, new: bool = False
) -> cp.ndarray:
    """
    Build smear matrix B for bp

    Parameters
    ----------
    f: cp.ndarray
        potential on nodes; shape (n_exc, n_pts)
    fb: cp.ndarray
        potential on adjacent electrodes; shape (n_exc, n_el)
    meas_pattern: cp.ndarray
        electrodes numbering pairs; shape (n_exc, n_meas, 2)
    new : bool, optional
        flag to use new matrices based computation, by default False.
        If `False` to smear-computation from ChabaneAmaury is used

    Returns
    -------
    cp.ndarray
        back-projection (smear) matrix; shape (n_exc, n_meas, n_pts), dtype= bool
    """
    if new:
        # new implementation not much faster! :(
        idx_meas_0 = meas_pattern[:, :, 0]
        idx_meas_1 = meas_pattern[:, :, 1]
        n_exc = meas_pattern.shape[0]  # number of excitations
        n_meas = meas_pattern.shape[1]  # number of measurements per excitations
        n_pts = f.shape[1]  # number of nodes
        idx_exc = cp.ones_like(idx_meas_0, dtype=int) * cp.arange(n_exc).reshape(
            n_exc, 1
        )
        f_min = cp.minimum(fb[idx_exc, idx_meas_0], fb[idx_exc, idx_meas_1])
        f_max = cp.maximum(fb[idx_exc, idx_meas_0], fb[idx_exc, idx_meas_1])
        # contruct matrices of shapes (n_exc, n_meas, n_pts) for comparison
        f_min = cp.repeat(f_min[:, :, cp.newaxis], n_pts, axis=2)
        f_max = cp.repeat(f_max[:, :, cp.newaxis], n_pts, axis=2)
        f0 = cp.repeat(f[:, :, cp.newaxis], n_meas, axis=2)
        f0 = f0.swapaxes(1, 2)
        return (f_min < f0) & (f0 <= f_max)
    else:
        # Replacing the below code by a faster implementation in Numpy
        def b_matrix_init(k):
            return _smear(f[k], fb[k], meas_pattern[k])

        return cp.array(list(map(b_matrix_init, cp.arange(f.shape[0]))))


def subtract_row(v: cp.ndarray, meas_pattern: cp.ndarray) -> cp.ndarray:
    """
    Build the voltage differences on axis=1 using the meas_pattern.
    v_diff[k] = v[i, :] - v[j, :]

    New implementation 33% less computation time

    Parameters
    ----------
    v: cp.ndarray
        Nx1 boundary measurements vector or NxM matrix; shape (n_exc,n_el,1)
    meas_pattern: cp.ndarray
        Nx2 subtract_row pairs; shape (n_exc, n_meas, 2)

    Returns
    -------
    cp.ndarray
        difference measurements v_diff
    """
    return v[meas_pattern[:, 0]] - v[meas_pattern[:, 1]]


def assemble(
    ke: cp.ndarray, tri: cp.ndarray, perm: cp.ndarray, n_pts: int, ref: int = 0
) -> csr_matrix:
    """
    Assemble the stiffness matrix (using sparse matrix)

    Parameters
    ----------
    ke: cp.ndarray
        n_tri x (n_dim x n_dim) 3d matrix
    tri: cp.ndarray
        the structure of mesh
    perm: cp.ndarray
        n_tri x 1 conductivities on elements
    n_pts: int
        number of nodes
    ref: int, optional
        reference electrode, by default 0

    Returns
    -------
    cp.ndarray
        NxN array of complex stiffness matrix

    Notes
    -----
    you may use sparse matrix (IJV) format to automatically add the local
    stiffness matrix to the global matrix.
    """
    n_tri, n_vertices = tri.shape

    # New: use IJV indexed sparse matrix to assemble K (fast, prefer)
    # index = cp.array([cp.meshgrid(no, no, indexing='ij') for no in tri])
    # note: meshgrid is slow, using handcraft sparse index, for example
    # let tri=[[1, 2, 3], [4, 5, 6]], then indexing='ij' is equivalent to
    # row = [1, 1, 1, 2, 2, 2, ...]
    # col = [1, 2, 3, 1, 2, 3, ...]
    row = cp.asarray(cp.repeat(tri, n_vertices).ravel())
    col = cp.asarray(cp.repeat(tri, n_vertices, axis=0).ravel())
    data = cp.asarray(cp.array([ke[i] * perm[i] for i in range(n_tri)]).ravel())

    # set reference nodes before constructing sparse matrix
    if 0 <= ref < n_pts:
        dirichlet_ind = cp.logical_or(row == ref, col == ref)
        # K[ref, :] = 0, K[:, ref] = 0
        row = row[~dirichlet_ind]
        col = col[~dirichlet_ind]
        data = data[~dirichlet_ind]
        # K[ref, ref] = 1.0
        row = cp.append(row, ref)
        col = cp.append(col, ref)
        data = cp.append(data, 1.0)

    # for efficient sparse inverse (csc)
    return sparse.csr_matrix((data, (row, col)), shape=(n_pts, n_pts))


def calculate_ke(pts: cp.ndarray, tri: cp.ndarray) -> cp.ndarray:
    """
    Calculate local stiffness matrix on all elements.

    Parameters
    ----------
    pts: cp.ndarray
        Nx2 (x,y) or Nx3 (x,y,z) coordinates of points
    tri: cp.ndarray
        Mx3 (triangle) or Mx4 (tetrahedron) connectivity of elements

    Returns
    -------
    cp.ndarray
        n_tri x (n_dim x n_dim) 3d matrix
    """
    n_tri, n_vertices = tri.shape

    # check dimension
    # '3' : triangles
    # '4' : tetrahedrons
    if n_vertices == 3:
        _k_local = _k_triangle
    elif n_vertices == 4:
        _k_local = _k_tetrahedron
    else:
        raise TypeError("The num of vertices of elements must be 3 or 4")

    # default data types for ke
    ke_array = cp.zeros((n_tri, n_vertices, n_vertices))
    for ei in range(n_tri):
        no = tri[ei, :]
        xy = pts[no]

        # compute the KIJ (permittivity=1.)
        ke = _k_local(xy)
        ke_array[ei] = ke

    return ke_array


def _k_triangle(xy: cp.ndarray) -> cp.ndarray:
    """
    Given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy: cp.ndarray
        (x,y) of nodes 1,2,3 given in counterclockwise manner

    Returns
    -------
    cp.ndarray
        local stiffness matrix
    """
    # edges (vector) of triangles
    s = xy[[2, 0, 1]] - xy[[1, 2, 0]]

    # area of triangles. Note, abs is removed since version 2020,
    # user must make sure all triangles are CCW (conter clock wised).
    # at = 0.5 * cp.linalg.det(s[[0, 1]])
    at = 0.5 * det2x2(s[0], s[1])

    # Local stiffness matrix (e for element)
    return cp.dot(s, s.T) / (4.0 * at)


def det2x2(s1: cp.ndarray, s2: cp.ndarray) -> float:
    """Calculate the determinant of a 2x2 matrix"""
    return s1[0] * s2[1] - s1[1] * s2[0]


def _k_tetrahedron(xy: cp.ndarray) -> cp.ndarray:
    """
    Given a point-matrix of an element, solving for Kij analytically
    using barycentric coordinates (simplex coordinates)

    Parameters
    ----------
    xy: cp.ndarray
        (x,y) of nodes 1, 2, 3, 4 given in counterclockwise manner, see notes.

    Returns
    -------
    cp.ndarray
        local stiffness matrix

    Notes
    -----
    A tetrahedron is described using [0, 1, 2, 3] (local node index) or
    [171, 27, 9, 53] (global index). Counterclockwise (CCW) is defined
    such that the barycentric coordinate of face (1->2->3) is positive.
    """
    s = xy[[2, 3, 0, 1]] - xy[[1, 2, 3, 0]]

    # volume of the tetrahedron, Note abs is removed since version 2020,
    # user must make sure all tetrahedrons are CCW (counter clock wised).
    vt = 1.0 / 6 * cla.det(s[[0, 1, 2]])

    # calculate area (vector) of triangle faces
    # re-normalize using alternative (+,-) signs
    ij_pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
    signs = [1, -1, 1, -1]
    a = cp.array([sign * cp.cross(s[i], s[j]) for (i, j), sign in zip(ij_pairs, signs)])

    # local (e for element) stiffness matrix
    return cp.dot(a, a.transpose()) / (36.0 * vt)


if __name__ == "__main__":
    """"""
