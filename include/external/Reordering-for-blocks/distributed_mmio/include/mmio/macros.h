#ifndef __MMIO_MACROS_H__
#define __MMIO_MACROS_H__

#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_LINE_LENGTH 1500
#define MM_MAX_TOKEN_LENGTH 64

typedef char MM_typecode[6];

/********************* MM_typecode query functions ***************************/

#define mm_is_matrix(typecode)	    ((typecode)[0]=='M')

#define mm_is_sparse(typecode)	    ((typecode)[1]=='C')
#define mm_is_coordinate(typecode)  ((typecode)[1]=='C')
#define mm_is_dense(typecode)	    ((typecode)[1]=='A')
#define mm_is_array(typecode)	    ((typecode)[1]=='A')

#define mm_is_complex(typecode)	    ((typecode)[2]=='C')
#define mm_is_real(typecode)		((typecode)[2]=='R')
#define mm_is_pattern(typecode)	    ((typecode)[2]=='P')
#define mm_is_integer(typecode)     ((typecode)[2]=='I')

#define mm_is_symmetric(typecode)   ((typecode)[3]=='S')
#define mm_is_general(typecode)	    ((typecode)[3]=='G')
#define mm_is_skew(typecode)	    ((typecode)[3]=='K')
#define mm_is_hermitian(typecode)   ((typecode)[3]=='H')

#define mm_get_idx_bytes(typecode)  ((uint8_t)((unsigned char)((typecode)[4])))
#define mm_get_val_bytes(typecode)  ((uint8_t)((unsigned char)((typecode)[5])))

int mm_is_valid(MM_typecode matcode);		/* too complex for a macro */


/********************* MM_typecode modify functions ***************************/

#define mm_set_matrix(typecode)	    ((*typecode)[0]='M')
#define mm_set_coordinate(typecode)	((*typecode)[1]='C')
#define mm_set_array(typecode)	    ((*typecode)[1]='A')
#define mm_set_dense(typecode)	    mm_set_array(typecode)
#define mm_set_sparse(typecode)	    mm_set_coordinate(typecode)

#define mm_set_complex(typecode)    ((*typecode)[2]='C')
#define mm_set_real(typecode)       ((*typecode)[2]='R')
#define mm_set_pattern(typecode)    ((*typecode)[2]='P')
#define mm_set_integer(typecode)    ((*typecode)[2]='I')

#define mm_set_symmetric(typecode)  ((*typecode)[3]='S')
#define mm_set_general(typecode)    ((*typecode)[3]='G')
#define mm_set_skew(typecode)	    ((*typecode)[3]='K')
#define mm_set_hermitian(typecode)  ((*typecode)[3]='H')

#define mm_set_idx_bytes(typecode, bytes)  ((*typecode)[4]=(char)((uint8_t)(bytes)))
#define mm_set_val_bytes(typecode, bytes)  ((*typecode)[5]=(char)((uint8_t)(bytes)))

#define mm_clear_typecode(typecode) ((*typecode)[0]=(*typecode)[1]=(*typecode)[2]=' ',(*typecode)[3]='G',(*typecode)[4]=(*typecode)[5]='0')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)


/********************* Matrix Market error codes ***************************/

#define MMIO_ERR_COULD_NOT_READ_FILE	11
#define MMIO_ERR_PREMATURE_EOF          12
#define MMIO_ERR_NOT_MTX				13
#define MMIO_ERR_NO_HEADER				14
#define MMIO_ERR_UNSUPPORTED_TYPE		15
#define MMIO_ERR_LINE_TOO_LONG			16
#define MMIO_ERR_COULD_NOT_WRITE_FILE	17
#define MMIO_ERR_SHOULD_NOT_HAPPEN		18


/******************** Matrix Market internal definitions ********************

   MM_matrix_typecode: 4-character sequence

				    ojbect 		sparse/   	data        storage 
						  		dense     	type        scheme

   string position:	 [0]        [1]			[2]         [3]

   Matrix typecode: M(atrix)  	C(oord)		R(eal)   	G(eneral)
						        A(array)	C(omplex) 	H(ermitian)
										    P(attern) 	S(ymmetric)
								            I(nteger) 	K(kew)

 ***********************************************************************/

#define MM_MTX_STR		      "matrix"
#define MM_ARRAY_STR	      "array"
#define MM_DENSE_STR	      "array"
#define MM_COORDINATE_STR     "coordinate" 
#define MM_SPARSE_STR	      "coordinate"
#define MM_COMPLEX_STR	      "complex"
#define MM_REAL_STR		      "real"
#define MM_INT_STR		      "integer"
#define MM_GENERAL_STR        "general"
#define MM_SYMM_STR		      "symmetric"
#define MM_HERM_STR		      "hermitian"
#define MM_SKEW_STR		      "skew-symmetric"
#define MM_PATTERN_STR        "pattern"


#endif // __MMIO_MACROS_H__