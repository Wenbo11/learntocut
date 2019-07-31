cimport cgurobi
cimport numpy as np
from libc.stdlib cimport malloc, free
import numpy

#5000 = 2000

def printerror(error, msg):
    if error:
        print(msg + ': error code', error)
        exit(1)

cdef class GurobiSolver(object):
    """
    a class that wraps gurobi solver state
    """

    cdef cgurobi.GRBenv* _gurobi_env
    cdef cgurobi.GRBmodel* _gurobi_model
    cdef int numvars
    cdef int numconstraints

    def __cinit__(self, int numvars):
        self.numvars = numvars
        self.numconstraints = 0 # initialize num of constraints

        # initialize an empty environment
        self._gurobi_env = NULL
        error = cgurobi.GRBloadenv(&self._gurobi_env, '')
        printerror(error, 'load env failed')

        # set simplex method
        cgurobi.GRBsetintparam(self._gurobi_env, 'Method', 1)

        # silence gurobi output
        cgurobi.GRBsetintparam(self._gurobi_env, 'OutputFlag', 0)

    def reset(self, np.ndarray c):

        # initialize model
        error = cgurobi.GRBnewmodel(self._gurobi_env, &self._gurobi_model, 'model', self.numvars,
                                    NULL, NULL, NULL, NULL, NULL)
        printerror(error, 'build new model failed')

        # ==== reset the IP problem ====
        # reset the objective c
        cdef int j
        for j in range(self.numvars):
            cgurobi.GRBsetdblattrelement(self._gurobi_model, 'Obj', j, c[j])
            cgurobi.GRBsetdblattrelement(self._gurobi_model, 'LB', j, 0.0)
            cgurobi.GRBsetdblattrelement(self._gurobi_model, 'UB', j, cgurobi.GRB_INFINITY )

        # add variables
        #for j in range(self.numvars):
        #    error = cgurobi.GRBaddvar(self._gurobi_model, 0, NULL, NULL, c[j], 0.0, cgurobi.GRB_INFINITY, cgurobi.GRB_CONTINUOUS, NULL)
        #     printerror(error))
        # model sense
        cgurobi.GRBsetintattr(self._gurobi_model, 'ModelSense', cgurobi.GRB_MINIMIZE)

        # initialize some parameters
        self.numconstraints = 0

    def add_cuts(self, np.ndarray index, np.ndarray a, np.ndarray b):
        # add one set of constraint
        cdef double* val = <double*>a.data
        cdef long* ind = <long*>index.data
        cdef int i
        cdef int* ind_
        try:
            ind_ = <int *> malloc(self.numvars * sizeof(int))
            for i in range(self.numvars):
                ind_[i] = ind[i]
                #print(ind_[i],val[i],b[0])
            error = cgurobi.GRBaddconstr(self._gurobi_model, self.numvars, ind_, val, cgurobi.GRB_LESS_EQUAL, b[0], NULL)
            printerror(error, 'add cuts failed')
        finally:
            free(ind_)
        self.numconstraints += 1

    def optimize(self):
        error = cgurobi.GRBoptimize(self._gurobi_model)
        printerror(error, 'optimize failed')

    def get_objval(self):
        cdef int status
        # check for optimization status
        cgurobi.GRBgetintattr(self._gurobi_model, <const char *>cgurobi.GRB_INT_ATTR_STATUS, &status)
        #print('status', status, cgurobi.GRB_OPTIMAL)
        if status == cgurobi.GRB_OPTIMAL:
            #print('optimization solved')
            error = cgurobi.GRBgetdblattr(self._gurobi_model, <const char *>cgurobi.GRB_DBL_ATTR_OBJVAL, &objval)
            printerror(error, 'get obj failed')
        else:
            objval = 0.0
        return objval, status

    def write_model(self):
        error = cgurobi.GRBwrite(self._gurobi_model, 'gurobimodel.lp')
        printerror(error, 'write model failed')

    def get_slack_solution(self):
        cdef double* x
        cdef int i
        try:
            x = <double *> malloc(self.numconstraints * sizeof(double))
            cgurobi.GRBgetdblattrarray(self._gurobi_model, 'Slack', 0, self.numconstraints, x)
            xfinal = numpy.zeros([self.numconstraints], dtype=numpy.float64)
            for i in range(self.numconstraints):
                xfinal[i] = x[i]
                #print(xfinal[i],x[i])
        finally:
            free(x)
        return xfinal

    def get_original_solution(self):
        cdef double* x
        cdef int i
        try:
            x = <double *> malloc(self.numvars * sizeof(double))
            cgurobi.GRBgetdblattrarray(self._gurobi_model, <const char *>cgurobi.GRB_DBL_ATTR_X, 0, self.numvars, x)
            xfinal = numpy.zeros([self.numvars], dtype=numpy.float64)
            for i in range(self.numvars):
                xfinal[i] = x[i]
                #print(xfinal[i],x[i])
        finally:
            free(x)
        return xfinal

    def get_invrow(self, int i):
        # return the ith row of B^-1 A
        cdef cgurobi.GRBsvec x
        cdef int j
        try:
            x.ind = <int *> malloc(5000 * sizeof(int))
            x.val = <double *> malloc(5000 * sizeof(double))
        
            error = cgurobi.GRBBinvRowi(self._gurobi_model, i, &x);
            printerror(error, 'get inv row failed')

            row = numpy.zeros([self.numvars + self.numconstraints], dtype=numpy.float64)
            for j in range(x.len):
                row[x.ind[j]] = x.val[j]
        finally:
            free(x.ind)
            free(x.val)
        return row

    def get_tableau(self):
        tab = numpy.zeros([self.numconstraints, self.numvars + self.numconstraints])
        cdef int i
        for i in range(self.numconstraints):
            tab[i,:] = self.get_invrow(i)
        return tab

    def get_basis(self):
        # get the binary vector that correponds to basic variables
        cdef int x[5000]
        cgurobi.GRBgetBasisHead(self._gurobi_model, x)
        xfinal = numpy.zeros([self.numconstraints], dtype=numpy.int32)
        for i in range(self.numconstraints):
            xfinal[i] = x[i]
            #print(xfinal[i],x[i])
        return xfinal

    # pickling
    def __getstate__(self):
        return {"_ezpickle_args" : self.numvars, "_ezpickle_kwargs": None}
    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)
