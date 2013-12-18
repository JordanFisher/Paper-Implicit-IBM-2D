#include"Python.h"
#include"arrayobject.h"

const double pi = 3.14159265358979323;
double Delta (double h, double r, int DeltaType) {
	if (DeltaType == 0)
	{
		if (abs(r) < 2 * h)
			return (1. + cos(pi * r / (2*h))) / (4. * h);
		else
			return 0;
	}
	else
	{
		double x = r / h;
		double absx = abs(x);
		if (absx <= 2)
			if (absx <= 1)
				return .125 * (3. - 2 * absx + sqrt(1. + 4. * absx - 4. * x * x)) / h;
			else
				return .125 * (5. - 2 * absx - sqrt(-7. + 12. * absx - 4 * x * x)) / h;
		else
			return 0;
	}
}


int mod (int a, int m) {
	return (a % m + m) % m;
}



// Put a matrix into EzSparse format
static PyObject* cEzSparse (PyObject *self, PyObject *args)
{
	PyArrayObject *A, *I, *Ii, *Id;
	int N, M, s, _s;
	int *ai, j, k, l;
	int *aStride, bStride, jStride;
	char* kStride;
	double *a;
	double sum;

	if (!PyArg_ParseTuple(args, "iiOOOO", &N, &M, &A, &I, &Ii, &Id)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check I array
	if (I->nd != 2 || I->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "I array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check Ii array
	if (Ii->nd != 2) {
		PyErr_SetString(PyExc_ValueError, "Ii array must be one-dimensional");

		return NULL;
	}

	for (j = 0; j < N; j++) {
		l = 0;
		for (k = 0; k < M; k++) {
			if (*(double *)(A->data + j*A->strides[0] + k*A->strides[1]) != 0.) {
				*(double *)(I->data + j*I->strides[0] + l*I->strides[1]) = 
					*(double *)(A->data + j*A->strides[0] + k*A->strides[1]);
				*(int *)(Ii->data + j*Ii->strides[0] + l*Ii->strides[1]) = k;
				l++;
			}
		}
		*(int *)(Id->data + j*Id->strides[0]) = l;
	}

	Py_INCREF(Py_None);
	return Py_None;
}


// Sparse matrix-matrix multiplication
static PyObject* cSparseMM (PyObject *self, PyObject *args)
{
	PyArrayObject *A, *B, *I, *Ii, *Id;
	int N, M, _M, s, _s;
	int *ai, j, k, l;
	int *aStride, bStride, jStride;
	char* kStride;
	double *a;
	double sum;

	if (!PyArg_ParseTuple(args, "iiiiOOOOO", &N, &M, &_M, &s, &I, &Ii, &Id, &A, &B)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check UField array
	if (A->nd != 2 || A->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "A array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check VField array
	if (B->nd != 2 || B->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "B array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check I array
	if (I->nd != 2 || I->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "I array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check Ii array
	if (Ii->nd != 2) {
		PyErr_SetString(PyExc_ValueError, "Ii array must be one-dimensional");

		return NULL;
	}

	// Make sure dimensions agree
	if (A->dimensions[0] != M || A->dimensions[1] != _M ||
		B->dimensions[0] != N || B->dimensions[1] != _M) {
		PyErr_SetString(PyExc_ValueError, "Error in dimensions");

		return NULL;
	}

	a = new double[s];
	ai = new int[s];
	aStride = new int[s];

	for (j = 0; j < N; j++) {
		for (l = 0; l < s; l++) {
			ai[l] = *(int *)(Ii->data + j*Ii->strides[0] + l*Ii->strides[1]);
			a[l] = *(double *)(I->data + j*I->strides[0] + l*I->strides[1]);
			aStride[l] = ai[l] * A->strides[0];
		}
		jStride = j * B->strides[0];

		_s = *(int *)(Id->data + j*Id->strides[0]);
		for (k = 0; k < _M; k++) {
			kStride = (A->data + k * A->strides[1]);
			sum = 0;
			for (l = 0; l < _s; l++)
				sum += a[l] * (*(double *)(aStride[l] + kStride));
			*(double *)(B->data + jStride + k*B->strides[1]) = sum;
		}
	}

	delete[] ai,a,aStride;

	Py_INCREF(Py_None);
	return Py_None;
}






static PyObject* cGaussSeidel (PyObject *self, PyObject *args)
{
	PyArrayObject *A, *b, *x;
	int N, i, j, Astr1, Astr2, xstr;
	double damp, S, Aii, bi;

	if (!PyArg_ParseTuple(args, "iOOOd", &N, &A, &b, &x, &damp)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check x array
	if (x->nd != 1 || x->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "x array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check b array
	if (b->nd != 1 || b->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "b array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check A array
	if (A->nd != 2 || A->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "A array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (x->dimensions[0] != N || b->dimensions[0] != N) {
		PyErr_SetString(PyExc_ValueError, "x, b input arrays must have same dimension");

		return NULL;
	}

	if (A->dimensions[0] != N || A->dimensions[1] != N) {
		PyErr_SetString(PyExc_ValueError, "A input matrix must have dimensions NxN");

		return NULL;
	}

	// Gauss-Seidel iterations
	Astr1 = A->strides[0];
	Astr2 = A->strides[1];
	xstr = x->strides[0];

	for (i = 0; i < N; i++) {
		S = 0.;
		for (j = 0; j < i; j++)
			S += (*(double *)(A->data + i*Astr1 + j*Astr2)) * (*(double *)(x->data + j*xstr));
		for (j = i+1; j < N; j++)
			S += (*(double *)(A->data + i*Astr1 + j*Astr2)) * (*(double *)(x->data + j*xstr));

		Aii = *(double *)(A->data + i*Astr1 + i*Astr2);
		bi = *(double *)(b->data + i*b->strides[0]);

		*(double *)(x->data + i * xstr) = (1.-damp) * (*(double *)(x->data + i * xstr)) +
			damp * (bi - S) / Aii;
	}

	Py_INCREF(Py_None);
	return Py_None;
}


static PyObject* cCollapseSum (PyObject *self, PyObject *args)
{
	PyArrayObject *x, *index, *f;
	int iN, xN, i, j;

	if (!PyArg_ParseTuple(args, "OOO", &x, &index, &f)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check x array
	if (x->nd != 1 || x->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "x array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check index array
	if (index->nd != 1) { // || index->descr->type_num != PyArray_INT) {
		PyErr_SetString(PyExc_ValueError, "index array must be one-dimensional and of type int (int64 in python)");

		return NULL;
	}

	// Type check f array
	if (f->nd != 1 || f->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "f array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	xN = x->dimensions[0];
	iN = index->dimensions[0];
	if (f->dimensions[0] != iN) {
		PyErr_SetString(PyExc_ValueError, "index, f input arrays must have same dimension");

		return NULL;
	}

	for (j = 0; j < iN; j++) {
		i = *(int *)(index->data + j*index->strides[0]);
		*(double *)(x->data + i * x->strides[0]) += *(double *)(f->data + j*f->strides[0]);
	}

	Py_INCREF(Py_None);
	return Py_None;
}


static PyObject* cComputeMatrix (PyObject *self, PyObject *args)
{
	PyArrayObject *X, *Y, *UField, *VField, *UField2, *VField2, *A;
	int N, M, Nb, dCutoff, j, k, _j, _k;
	double h, hb;
	double a, b, a1b1, a0b1, a1b0, a0b0;
	int abs1, abs2;

	if (!PyArg_ParseTuple(args, "OOiiiddOOOOOi", &X, &Y, &N, &M, &Nb, &h, &hb, &UField, &VField, &UField2, &VField2, &A, &dCutoff)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check X array
	if (X->nd != 1 || X->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "X array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check Y array
	if (Y->nd != 1 || Y->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "Y array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check UField array
	if (UField->nd != 2 || UField->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "UField array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check VField array
	if (VField->nd != 2 || VField->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "VField array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check A array
	if (A->nd != 2 || A->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "A array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (X->dimensions[0] != Nb || Y->dimensions[0] != Nb) {
		PyErr_SetString(PyExc_ValueError, "X, Y input arrays must have dim Nb");

		return NULL;
	}

	if (UField->dimensions[0] != N || UField->dimensions[1] != M ||
		VField->dimensions[0] != N || VField->dimensions[1] != M) {
		PyErr_SetString(PyExc_ValueError, "UField, VField input arrays must have dim NxM");

		return NULL;
	}

	if (A->dimensions[0] != 2*Nb || A->dimensions[1] != 2*Nb) {
		PyErr_SetString(PyExc_ValueError, "A input array must have dim 2*Nbx2*Nb");

		return NULL;
	}

	if (dCutoff <= 0) dCutoff = 1000000;

	for (j = 0; j < Nb; j++) {
		for (k = 0; k < Nb; k++) {
			a = (*(double *)(X->data + j*X->strides[0]) - *(double *)(X->data + k*X->strides[0])) / h;
			b = (*(double *)(Y->data + j*Y->strides[0]) - *(double *)(Y->data + k*Y->strides[0])) / h;

			_j = (int) floor(a);
			_k = (int) floor(b);

			abs1 = abs(_j%N);
			abs2 = abs(_k%M);
			if ((abs1 < dCutoff || abs1 > N-dCutoff) && (abs2 < dCutoff || abs2 > M-dCutoff)) {
				a -= (double) _j;
				b -= (double) _k;
				_j = _j + N;
				_k = _k + M;

				a1b1 = (1. - a) * (1. - b) * hb;
				a1b0 = (1. - a) * b * hb;
				a0b1 = a * (1. - b) * hb;
				a0b0 = a * b * hb;

				if (j >= k) {
					*(double *)(A->data + j*A->strides[0] + k*A->strides[1]) =
						a1b1 * (*(double *)(UField->data + (_j%N)*UField->strides[0] + (_k%M)*UField->strides[1])) +
						a0b1 * (*(double *)(UField->data + ((_j+1)%N)*UField->strides[0] + (_k%M)*UField->strides[1])) +
						a1b0 * (*(double *)(UField->data + (_j%N)*UField->strides[0] + ((_k+1)%M)*UField->strides[1])) +
						a0b0 * (*(double *)(UField->data + ((_j+1)%N)*UField->strides[0] + ((_k+1)%M)*UField->strides[1]));
					*(double *)(A->data + (j+Nb)*A->strides[0] + (k+Nb)*A->strides[1]) = 
						a1b1 * (*(double *)(VField2->data + (_j%N)*VField2->strides[0] + (_k%M)*VField2->strides[1])) +
						a1b0 * (*(double *)(VField2->data + (_j%N)*VField2->strides[0] + ((_k+1)%M)*VField2->strides[1])) +
						a0b1 * (*(double *)(VField2->data + ((_j+1)%N)*VField2->strides[0] + (_k%M)*VField2->strides[1])) +
						a0b0 * (*(double *)(VField2->data + ((_j+1)%N)*VField2->strides[0] + ((_k+1)%M)*VField2->strides[1]));

					*(double *)(A->data + k*A->strides[0] + j*A->strides[1]) = 
						*(double *)(A->data + j*A->strides[0] + k*A->strides[1]);
					*(double *)(A->data + (k+Nb)*A->strides[0] + (j+Nb)*A->strides[1]) = 
						*(double *)(A->data + (j+Nb)*A->strides[0] + (k+Nb)*A->strides[1]);
				}

				*(double *)(A->data + (j+Nb)*A->strides[0] + k*A->strides[1]) =
					a1b1 * (*(double *)(VField->data + (_j%N)*VField->strides[0] + (_k%M)*VField->strides[1])) +
					a0b1 * (*(double *)(VField->data + ((_j+1)%N)*VField->strides[0] + (_k%M)*VField->strides[1])) +
					a1b0 * (*(double *)(VField->data + (_j%N)*VField->strides[0] + ((_k+1)%M)*VField->strides[1])) +
					a0b0 * (*(double *)(VField->data + ((_j+1)%N)*VField->strides[0] + ((_k+1)%M)*VField->strides[1]));

				*(double *)(A->data + k*A->strides[0] + (j+Nb)*A->strides[1]) =
					*(double *)(A->data + (j+Nb)*A->strides[0] + k*A->strides[1]);

					
			}
		}
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* cComputeMatrixDirect (PyObject *self, PyObject *args)
{
	PyArrayObject *X, *Y, *UField, *VField, *UField2, *VField2, *A;
	int N, M, Nb, dCutoff, j, k, _j, _k;
	double h, hb;
	double a, b, a1b1, a0b1, a1b0, a0b0;
	int abs1, abs2;

	if (!PyArg_ParseTuple(args, "OOiiiddOOOOOi", &X, &Y, &N, &M, &Nb, &h, &hb, &UField, &VField, &UField2, &VField2, &A, &dCutoff)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check X array
	if (X->nd != 1 || X->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "X array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check Y array
	if (Y->nd != 1 || Y->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "Y array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check UField array
	if (UField->nd != 2 || UField->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "UField array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check VField array
	if (VField->nd != 2 || VField->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "VField array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check A array
	if (A->nd != 2 || A->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "A array must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (X->dimensions[0] != Nb || Y->dimensions[0] != Nb) {
		PyErr_SetString(PyExc_ValueError, "X, Y input arrays must have dim Nb");

		return NULL;
	}

	if (UField->dimensions[0] != N || UField->dimensions[1] != M ||
		VField->dimensions[0] != N || VField->dimensions[1] != M) {
		PyErr_SetString(PyExc_ValueError, "UField, VField input arrays must have dim NxM");

		return NULL;
	}

	if (A->dimensions[0] != 2*Nb || A->dimensions[1] != 2*Nb) {
		PyErr_SetString(PyExc_ValueError, "A input array must have dim 2*Nbx2*Nb");

		return NULL;
	}

	if (dCutoff <= 0) dCutoff = 1000000;

	double div = 1. / h;
	a = 0;
	b = 0;
	for (j = 0; j < Nb; j++) {
		for (k = 0; k < Nb; k++) {
			//a = (*(double *)(X->data + j*X->strides[0]) - *(double *)(X->data + k*X->strides[0])) * div;
			//b = (*(double *)(Y->data + j*Y->strides[0]) - *(double *)(Y->data + k*Y->strides[0])) * div;

			_j = (int) floor(a);
			_k = (int) floor(b);

				_j = _j + N;
				_k = _k + M;

				if (j >= k) {
					*(double *)(A->data + j*A->strides[0] + k*A->strides[1]) =
						(*(double *)(UField->data + (_j%N)*UField->strides[0] + (_k%M)*UField->strides[1]));
					*(double *)(A->data + (j+Nb)*A->strides[0] + (k+Nb)*A->strides[1]) = 
						(*(double *)(VField2->data + (_j%N)*VField2->strides[0] + (_k%M)*VField2->strides[1]));
				}

				*(double *)(A->data + (j+Nb)*A->strides[0] + k*A->strides[1]) =
					(*(double *)(VField->data + (_j%N)*VField->strides[0] + (_k%M)*VField->strides[1]));

				*(double *)(A->data + k*A->strides[0] + (j+Nb)*A->strides[1]) =
					*(double *)(A->data + (j+Nb)*A->strides[0] + k*A->strides[1]);					
		}
	}

	Py_INCREF(Py_None);
	return Py_None;
}



static PyObject* cCentralDerivative_x (PyObject *self, PyObject *args)
{
	PyArrayObject *f, *df;
	int N, M, j, k;
	double h, h2;

	if (!PyArg_ParseTuple(args, "iidOO", &N, &M, &h, &f, &df)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check f array
	if (f->nd != 2 || f->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "f array must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check df array
	if (df->nd != 2 || df->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "df array must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (f->dimensions[0] != N || f->dimensions[1] != M ||
		df->dimensions[0] != N || df->dimensions[1] != M) {
		PyErr_SetString(PyExc_ValueError, "f, df input arrays must have dim NxM");

		return NULL;
	}
	
	h2 = 2. * h;
	for (k = 0; k < M; k++) {
		for (j = 1; j < N-1; j++) {
			*(double *)(df->data + j*df->strides[0] + k*df->strides[1]) =
				(*(double *)(f->data + (j+1)*f->strides[0] + k*f->strides[1]) -
				*(double *)(f->data + (j-1)*f->strides[0] + k*f->strides[1])) / h2;
//            df[i][j] = (f[i+1][j] - f[i-1][j]) / (2. * h)
		}
		*(double *)(df->data + 0*df->strides[0] + k*df->strides[1]) =
			(*(double *)(f->data + 1*f->strides[0] + k*f->strides[1]) -
			*(double *)(f->data + (N-1)*f->strides[0] + k*f->strides[1])) / h2;
		*(double *)(df->data + (N-1)*df->strides[0] + k*df->strides[1]) =
			(*(double *)(f->data + 0*f->strides[0] + k*f->strides[1]) -
			*(double *)(f->data + (N-2)*f->strides[0] + k*f->strides[1])) / h2; 
//        df[0][j] = (f[1][j] - f[N-1][j]) / (2. * h)
//        df[N-1][j] = (f[0][j] - f[N-2][j]) / (2. * h)
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* cCentralDerivative_y (PyObject *self, PyObject *args)
{
	PyArrayObject *f, *df;
	int N, M, j, k;
	double h, h2;

	if (!PyArg_ParseTuple(args, "iidOO", &N, &M, &h, &f, &df)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check f array
	if (f->nd != 2 || f->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "f array must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check df array
	if (df->nd != 2 || df->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "df array must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (f->dimensions[0] != N || f->dimensions[1] != M ||
		df->dimensions[0] != N || df->dimensions[1] != M) {
		PyErr_SetString(PyExc_ValueError, "f, df input arrays must have dim NxM");

		return NULL;
	}
	
	h2 = 2. * h;	 
	for (j = 0; j < N; j++) {
		for (k = 1; k < M-1; k++) {
			*(double *)(df->data + j*df->strides[0] + k*df->strides[1]) =
				(*(double *)(f->data + j*f->strides[0] + (k+1)*f->strides[1]) -
				*(double *)(f->data + j*f->strides[0] + (k-1)*f->strides[1])) / h2;
//            df[i][j] = (f[i][j+1] - f[i][j-1]) / (2. * h)
		}
		*(double *)(df->data + j*df->strides[0] + 0*df->strides[1]) =
			(*(double *)(f->data + j*f->strides[0] + 1*f->strides[1]) -
			*(double *)(f->data + j*f->strides[0] + (M-1)*f->strides[1])) / h2;
		*(double *)(df->data + j*df->strides[0] + (M-1)*df->strides[1]) =
			(*(double *)(f->data + j*f->strides[0] + 0*f->strides[1]) -
			*(double *)(f->data + j*f->strides[0] + (M-2)*f->strides[1])) / h2; 
//        df[i][0] = (f[i][1] - f[i][M-1]) / (2. * h)
//        df[i][M-1] = (f[i][0] - f[i][M-2]) / (2. * h)
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* cInitWideLaplacian (PyObject *self, PyObject *args)
{
	PyArrayObject *Lambda;
	int N, M, j, k;
	double h, h2;

	if (!PyArg_ParseTuple(args, "iidO", &N, &M, &h, &Lambda)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check Lambda array
	if (Lambda->nd != 2 || Lambda->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "Lambda array must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (Lambda->dimensions[0] != N || Lambda->dimensions[1] != M) {
		PyErr_SetString(PyExc_ValueError, "Lambda input array must have dim NxM");

		return NULL;
	}

	h2 = pow(h,2.);
	for (j = 0; j < N; j++) {
		for (k = 0; k < M; k++) {
			*(double *)(Lambda->data + j*Lambda->strides[0] + k*Lambda->strides[1]) =
				(double) -(pow(sin(2.*pi*double(j) / double(N)),2.) + pow(sin(2.*pi*double(k) / double(M)),2.)) / h2;
//            Lambda[j,k] = -sin(2*pi*j / N)**2 / h**2 - sin(2*pi*k / M)**2 / h**2

            if (*(double *)(Lambda->data + j*Lambda->strides[0] + k*Lambda->strides[1]) == 0)
                *(double *)(Lambda->data + j*Lambda->strides[0] + k*Lambda->strides[1]) = 1;
		}
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* cInitShortLaplacian (PyObject *self, PyObject *args)
{
	PyArrayObject *Lambda;
	int N, M, j, k;
	double h, h2;

	if (!PyArg_ParseTuple(args, "iidO", &N, &M, &h, &Lambda)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check Lambda array
	if (Lambda->nd != 2 || Lambda->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "Lambda array must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (Lambda->dimensions[0] != N || Lambda->dimensions[1] != M) {
		PyErr_SetString(PyExc_ValueError, "Lambda input array must have dim NxM");

		return NULL;
	}

	h2 = pow(h,2.);
	for (j = 0; j < N; j++) {
		for (k = 0; k < M; k++) {
			*(double *)(Lambda->data + j*Lambda->strides[0] + k*Lambda->strides[1]) =
				(double) (2. * cos(2.*pi*double(j)/double(N)) + 2. * cos(2.*pi*double(k)/double(M)) - 4.) / h2;
//			  Lambda[j,k] = (2 * cos(2.*pi*j/N) - 2.) / h**2 + (2 * cos(2.*pi*k/N) - 2.) / h**2
            if (*(double *)(Lambda->data + j*Lambda->strides[0] + k*Lambda->strides[1]) == 0)
                *(double *)(Lambda->data + j*Lambda->strides[0] + k*Lambda->strides[1]) = 1;
		}
	}

	Py_INCREF(Py_None);
	return Py_None;
}


static PyObject* cInitDxSymbol (PyObject *self, PyObject *args)
{
	PyArrayObject *DxSymbol;
	int N, M, j, k;
	double h;

	if (!PyArg_ParseTuple(args, "iidO", &N, &M, &h, &DxSymbol)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check DxSymbol array
	if (DxSymbol->nd != 2 || DxSymbol->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "DxSymbol array must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (DxSymbol->dimensions[0] != N || DxSymbol->dimensions[1] != M) {
		PyErr_SetString(PyExc_ValueError, "DxSymbol input array must have dim NxM");

		return NULL;
	}

	for (j = 0; j < N; j++) 
		for (k = 0; k < M; k++) 
			*(double *)(DxSymbol->data + j*DxSymbol->strides[0] + k*DxSymbol->strides[1]) =
				(double) sin(2. * pi * double(j) / double(N)) / h;
// 			  DxSymbol[j,k] = 1j * sin(2*pi*j / N) / h

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* cInitDySymbol (PyObject *self, PyObject *args)
{
	PyArrayObject *DySymbol;
	int N, M, j, k;
	double h;

	if (!PyArg_ParseTuple(args, "iidO", &N, &M, &h, &DySymbol)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check DySymbol array
	if (DySymbol->nd != 2 || DySymbol->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "DySymbol array must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (DySymbol->dimensions[0] != N || DySymbol->dimensions[1] != M) {
		PyErr_SetString(PyExc_ValueError, "DySymbol input array must have dim NxM");

		return NULL;
	}

	for (j = 0; j < N; j++) 
		for (k = 0; k < M; k++) 
			*(double *)(DySymbol->data + j*DySymbol->strides[0] + k*DySymbol->strides[1]) =
				(double) sin(2.*pi*k / double(M)) / h;
// 			  DxSymbol[j,k] = 1j * sin(2*pi*k / M) / h

	Py_INCREF(Py_None);
	return Py_None;
}




static PyObject* cWholeGridSpread (PyObject *self, PyObject *args)
{
	PyArrayObject *U, *u;
	int n, m, j, k, _j, _k, r;
	double h, _h, delt, deltx, x, y, _h2;
	int DeltaType;

	if (!PyArg_ParseTuple(args, "OddiOi", &u, &h, &_h, &r, &U, &DeltaType)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check u array
	if (u->nd != 2 || u->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "u array must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check U array
	if (U->nd != 2 || U->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "U array must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}
	
	_h2 = _h * _h;

	n = U->dimensions[0];
	m = U->dimensions[1];

	// Make sure dimensions agree
	if (u->dimensions[0] != n || u->dimensions[1] != m) {
		PyErr_SetString(PyExc_ValueError, "u, U input arrays must have same shape");

		return NULL;
	}

	for (_j = 0; _j < n; _j++) {
		for (_k = 0; _k < m; _k++) {
			x = _j * _h;
			y = _k * _h;

			for (j = _j - r; j <= _j + r; j++) {
				deltx = Delta(h, j * _h - x, DeltaType);
				for (k = _k - r; k <= _k + r; k++) {
					delt = deltx * Delta(h, k * _h - y, DeltaType) * _h2;
					*(double *)(U->data + _j*U->strides[0] + _k*U->strides[1]) += 
						*(double *)(u->data + ((j+n)%n)*u->strides[0] + ((k+m)%m)*u->strides[1]) * delt;
				}
			}
		}
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* cForceToGrid (PyObject *self, PyObject *args)
{
	PyArrayObject *X, *Y, *Fx, *Fy, *fx, *fy;
	int n, m, Nb, i, j, k, jMin, jMax, kMin, kMax;
	double h, hb, delt, deltx;
	int DeltaType;

	if (!PyArg_ParseTuple(args, "iididOOOOOOi", &n, &m, &h, &Nb, &hb, &X, &Y, &Fx, &Fy, &fx, &fy, &DeltaType)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check X,Y,Fx,Fy arrays
	if (X->nd != 1 || X->descr->type_num != PyArray_DOUBLE || 
		Y->nd != 1 || Y->descr->type_num != PyArray_DOUBLE ||
		Fx->nd != 1 || Fx->descr->type_num != PyArray_DOUBLE || 
		Fy->nd != 1 || Fy->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "X,Y,Fx,Fy arrays must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check fx,fy array
	if (fx->nd != 2 || fx->descr->type_num != PyArray_DOUBLE ||
		fy->nd != 2 || fy->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "fx,fy arrays must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (fx->dimensions[0] != n || fx->dimensions[1] != m ||
		fy->dimensions[0] != n || fy->dimensions[1] != m) {
		PyErr_SetString(PyExc_ValueError, "fx, fy input arrays must have dim NxM");

		return NULL;
	}

	if (X->dimensions[0] != Nb || Y->dimensions[0] != Nb ||
		Fx->dimensions[0] != Nb || Fy->dimensions[0] != Nb) {
		PyErr_SetString(PyExc_ValueError, "X,Y,Fx,Fy input arrays must have dim Nb");

		return NULL;
	}

	for (j = 0; j < n; j++) {
		for (k = 0; k < m; k++) {
			*(double *)(fx->data + j*fx->strides[0] + k*fx->strides[1]) = 0.;
			*(double *)(fy->data + j*fy->strides[0] + k*fy->strides[1]) = 0.;
		}
	}

	for (i = 0; i < Nb; i++) {
		jMin = int(*(double *)(X->data + i*X->strides[0]) / h - 2.);
		jMax = jMin + 5;
		kMin = int(*(double *)(Y->data + i*Y->strides[0]) / h - 2.);
		kMax = kMin + 5;	
	
		for (j = jMin; j <= jMax; j++) {
			deltx = Delta(h, j * h - *(double *)(X->data + i*X->strides[0]), DeltaType);
			for (k = kMin; k <= kMax; k++) {
				delt = deltx * Delta(h, k * h - *(double *)(Y->data + i*Y->strides[0]), DeltaType) * hb;

				*(double *)(fx->data + mod(j,n)*fx->strides[0] + mod(k,m)*fx->strides[1]) += 
					*(double *)(Fx->data + i*Fx->strides[0]) * delt;
				*(double *)(fy->data + mod(j,n)*fy->strides[0] + mod(k,m)*fy->strides[1]) += 
					*(double *)(Fy->data + i*Fy->strides[0]) * delt;
			}
		}
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* cVelToFiber (PyObject *self, PyObject *args)
{
	PyArrayObject *X, *Y, *u, *v, *Xvel, *Yvel;
	int n, m, Nb, i, j, k, jMin, jMax, kMin, kMax;
	double h, h2, hb, dt, delt, deltx;
	int DeltaType;

	if (!PyArg_ParseTuple(args, "iididdOOOOOOi", &n, &m, &h, &Nb, &hb, &dt, &X, &Y, &u, &v, &Xvel, &Yvel, &DeltaType)) {
		PyErr_SetString(PyExc_ValueError, "yo, some funky shit went down from here to c, best make sure you got your type set straight");

		return NULL;
	}

	// Type check X,Y,Xvel,Yvel arrays
	if (X->nd != 1 || X->descr->type_num != PyArray_DOUBLE || 
		Y->nd != 1 || Y->descr->type_num != PyArray_DOUBLE ||
		Xvel->nd != 1 || Xvel->descr->type_num != PyArray_DOUBLE || 
		Yvel->nd != 1 || Yvel->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "X,Y,Xvel,Yvel arrays must be one-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Type check u,v array
	if (u->nd != 2 || u->descr->type_num != PyArray_DOUBLE ||
		v->nd != 2 || v->descr->type_num != PyArray_DOUBLE) {
		PyErr_SetString(PyExc_ValueError, "u,v arrays must be two-dimensional and of type double (float64 in python)");

		return NULL;
	}

	// Make sure dimensions agree
	if (u->dimensions[0] != n || u->dimensions[1] != m ||
		v->dimensions[0] != n || v->dimensions[1] != m) {
		PyErr_SetString(PyExc_ValueError, "u,v input arrays must have dim NxM");

		return NULL;
	}

	if (X->dimensions[0] != Nb || Y->dimensions[0] != Nb ||
		Xvel->dimensions[0] != Nb || Yvel->dimensions[0] != Nb) {
		PyErr_SetString(PyExc_ValueError, "X,Y,Xvel,Yvel input arrays must have dim Nb");

		return NULL;
	}

	for (i = 0; i < Nb; i++) {
		*(double *)(Xvel->data + i*Xvel->strides[0]) = 0.;
		*(double *)(Yvel->data + i*Yvel->strides[0]) = 0.;
	}

	h2 = pow(h,2.);

	for (i = 0; i < Nb; i++) {
		jMin = int(*(double *)(X->data + i*X->strides[0]) / h - 2.);
		jMax = jMin + 5;
		kMin = int(*(double *)(Y->data + i*Y->strides[0]) / h - 2.);
		kMax = kMin + 5;	
	
		for (j = jMin; j <= jMax; j++) {
			deltx = Delta(h, j * h - *(double *)(X->data + i*X->strides[0]), DeltaType);
			for (k = kMin; k <= kMax; k++) {
				delt = deltx * Delta(h, k * h - *(double *)(Y->data + i*Y->strides[0]), DeltaType) * h2;

				*(double *)(Xvel->data + i*Xvel->strides[0]) += 
					*(double *)(u->data + mod(j,n)*u->strides[0] + mod(k,m)*u->strides[1]) * delt;
				*(double *)(Yvel->data + i*Yvel->strides[0]) += 
					*(double *)(v->data + mod(j,n)*v->strides[0] + mod(k,m)*v->strides[1]) * delt;
			}
		}
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyMethodDef IB_c_methods[] = {
	{"EzSparse", cEzSparse, METH_VARARGS, "EzSparse (N, M, A, I, Ii, Id)"},
	{"SparseMM", cSparseMM, METH_VARARGS, "SparseMM (N, M, s, I, Ii, Id, A, B)"},
	{"GaussSeidel", cGaussSeidel, METH_VARARGS, "GaussSeidel (N, A, b, x, damp)"},
	{"CollapseSum", cCollapseSum, METH_VARARGS, "CollapseSum (x, index, f)"},
	{"WholeGridSpread", cWholeGridSpread, METH_VARARGS, "WholeGridSpread (u, v, h, _h, r, Fx, Fy)"},
	{"InitWideLaplacian", cInitWideLaplacian, METH_VARARGS, "InitWideLaplacian (N, M, h, Lambda)"},
	{"InitShortLaplacian", cInitShortLaplacian, METH_VARARGS, "InitShortLaplacian (N, M, h, Lambda)"},
	{"InitDxSymbol", cInitDxSymbol, METH_VARARGS, "InitDxSymbol (N, M, h, DxSymbol)"},
	{"InitDySymbol", cInitDySymbol, METH_VARARGS, "InitDySymbol (N, M, h, DySymbol)"},
	{"CentralDerivative_x", cCentralDerivative_x, METH_VARARGS, "CentralDerivative_x (N, M, h, f, df)"},
	{"CentralDerivative_y", cCentralDerivative_y, METH_VARARGS, "CentralDerivative_y (N, M, h, f, df)"},
	{"ComputeMatrix", cComputeMatrix, METH_VARARGS, "ComputeMatrix (X, Y, N, M, Nb, h, hb, UField, VField, A, dCutoff)"},
	{"ComputeMatrixDirect", cComputeMatrixDirect, METH_VARARGS, "ComputeMatrixDirect (X, Y, N, M, Nb, h, hb, UField, VField, A, dCutoff)"},
	{"ForceToGrid", cForceToGrid, METH_VARARGS, "ForceToGrid (N, M, h, Nb, hb, X, Y, Boundary_fx, Boundary_fy, fx, fy)"},
	{"VelToFiber", cVelToFiber, METH_VARARGS, "VelToFiber (N, M, h, Nb, hb, dt, X, Y, u, v)"},
	{NULL, NULL}
};

PyMODINIT_FUNC
initIB_c(void)
{
	Py_InitModule("IB_c", IB_c_methods);
}
