def create_matrix(rows, cols, fill_value=0.0):
  return [[fill_value for _ in range(cols)] for _ in range(rows)]

def dot(matrix_a, matrix_b):
  rows_a = len(matrix_a)
  cols_a = len(matrix_a[0])
  rows_b = len(matrix_b)
  cols_b = len(matrix_b[0])
  
  if cols_a != rows_b:
    raise ValueError("You failed: matrices dimentions mismatch for dot product")
  
  result = create_matrix(rows_a, cols_b)
  for i in range(rows_a):
    for j in range(cols_b):
      for k in range(cols_a):
        result[i][j] += matrix_a[i][k] * matrix_b[k][j]
  return result

def transpose(matrix):
  rows = len(matrix)
  cols = len(matrix[0])
  result = create_matrix(cols, rows)
  for i in range(rows):
    for j in range(cols):
      result[j][i] = matrix[i][j]
  return result

def add_matrices(matrix_a, matrix_b):
  rows = len(matrix_a)
  cols = len(matrix_a[0])
  result = create_matrix(rows, cols)
  for i in range(rows):
    for j in range(cols):
      result[i][j] = matrix_a[i][j] + matrix_b[i][j]
  return result

def subtract_matrices(matrix_a, matrix_b):
  rows = len(matrix_a)
  cols = len(matrix_a[0])
  result = create_matrix(rows, cols)
  for i in range(rows):
    for j in range(cols):
      result[i][j] = matrix_a[i][j] - matrix_b[i][j]
  return result

def multiply_scalar_matrix(matrix, scalar):
  rows = len(matrix)
  cols = len(matrix[0])
  result = create_matrix(rows, cols)
  for i in range(rows):
    for j in range(cols):
      result[i][j] = matrix[i][j] * scalar
  return result

def multiply_matrices_elementwise(matrix_a, matrix_b):
  rows = len(matrix_a)
  cols = len(matrix_a[0])
  result = create_matrix(rows, cols)
  for i in range(rows):
    for j in range(cols):
      result[i][j] = matrix_a[i][j] * matrix_b[i][j]
  return result

def apply_function(matrix, func):
  rows = len(matrix)
  cols = len(matrix[0])
  result = create_matrix(rows, cols)
  for i in range(rows):
    for j in range(cols):
      result[i][j] = func(matrix[i][j])
  return result