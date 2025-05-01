import numpy as np

class MatrizInvalidaError(Exception):
    """Excepción personalizada para errores de matrices inválidas."""
    pass

class Matriz:
    """Clase para representar y operar matrices."""
    
    def __init__(self, datos):
        """Inicializa la matriz después de validar los datos."""
        if isinstance(datos, np.ndarray):
            self.datos = datos
        else:
            raise MatrizInvalidaError("Los datos deben ser un array de numpy.")
        self.validar()

    @classmethod
    def aleatoria(cls, filas, columnas, minimo=0, maximo=10, tipo=int):
        """Genera una matriz aleatoria."""
        if filas <= 0 or columnas <= 0:
            raise ValueError("Filas y columnas deben ser positivas.")
        if tipo == int:
            datos = np.random.randint(minimo, maximo, size=(filas, columnas))
        elif tipo == float:
            datos = np.random.uniform(minimo, maximo, size=(filas, columnas))
        else:
            raise TypeError("Tipo no soportado. Usa int o float.")
        return cls(datos)

    def validar(self):
        """Verifica que la matriz sea válida."""
        if self.datos is None:
            raise MatrizInvalidaError("La matriz no puede ser None.")
        if not isinstance(self.datos, np.ndarray):
            raise MatrizInvalidaError("La matriz debe ser un objeto numpy.ndarray.")
        if self.datos.size == 0:
            raise MatrizInvalidaError("La matriz no puede estar vacía.")
        if np.isnan(self.datos).any():
            raise MatrizInvalidaError("La matriz contiene valores nulos o NaN.")

    def suma(self, otra):
        """Suma dos matrices."""
        if self.datos.shape != otra.datos.shape:
            raise ValueError("Las matrices deben tener la misma dimensión para sumarse.")
        return Matriz(self.datos + otra.datos)

    def resta(self, otra):
        """Resta dos matrices."""
        if self.datos.shape != otra.datos.shape:
            raise ValueError("Las matrices deben tener la misma dimensión para restarse.")
        return Matriz(self.datos - otra.datos)

    def multiplicacion(self, otra):
        """Multiplica dos matrices."""
        if self.datos.shape[1] != otra.datos.shape[0]:
            raise ValueError("Dimensiones incompatibles para la multiplicación.")
        return Matriz(np.matmul(self.datos, otra.datos))

    def inverso_aditivo(self):
        """Retorna el inverso aditivo de la matriz."""
        return Matriz(-self.datos)

    def identidad_aditiva(self):
        """Retorna la matriz identidad aditiva (matriz de ceros)."""
        return Matriz(np.zeros_like(self.datos))

    # Propiedades
    def es_conmutativa(self, otra):
        """Verifica si la suma es conmutativa."""
        return np.allclose(self.datos + otra.datos, otra.datos + self.datos)

    def es_asociativa(self, B, C):
        """Verifica si la suma es asociativa."""
        izquierda = (self.datos + B.datos) + C.datos
        derecha = self.datos + (B.datos + C.datos)
        return np.allclose(izquierda, derecha)

    def es_distributiva(self, B, C):
        """Verifica la propiedad distributiva de la multiplicación."""
        izquierda = np.matmul(self.datos, (B.datos + C.datos))
        derecha = np.matmul(self.datos, B.datos) + np.matmul(self.datos, C.datos)
        return np.allclose(izquierda, derecha)

    def verifica_inverso(self):
        """Verifica que el inverso aditivo funcione correctamente."""
        inverso = self.inverso_aditivo()
        suma = self.datos + inverso.datos
        return np.allclose(suma, np.zeros_like(self.datos))

    def verifica_identidad(self):
        """Verifica que la identidad aditiva funcione correctamente."""
        identidad = self.identidad_aditiva()
        suma = self.datos + identidad.datos
        return np.allclose(suma, self.datos)

    def __str__(self):
        """Representación bonita de la matriz."""
        return str(self.datos)

# --------------------------------------------------------------
# Pruebas de funcionamiento
# --------------------------------------------------------------

if __name__ == "__main__":
    # Generar matrices de ejemplo
    A = Matriz.aleatoria(3, 3)
    B = Matriz.aleatoria(3, 3)
    C = Matriz.aleatoria(3, 3)

    print("Matriz A:\n", A)
    print("\nMatriz B:\n", B)
    print("\nMatriz C:\n", C)

    # Propiedades aritméticas
    print("\n¿Conmutativa A + B = B + A?", A.es_conmutativa(B))
    print("¿Asociativa (A+B)+C = A+(B+C)?", A.es_asociativa(B, C))

    try:
        print("¿Distributiva A*(B+C) = A*B + A*C?", A.es_distributiva(B, C))
    except ValueError as e:
        print("Error en distributiva:", e)

    # Inverso aditivo
    print("\n¿Inverso aditivo correcto (A + (-A) = 0)?", A.verifica_inverso())

    # Identidad aditiva
    print("¿Identidad aditiva correcta (A + 0 = A)?", A.verifica_identidad())

    # Operaciones básicas
    suma_AB = A.suma(B)
    print("\nSuma A + B:\n", suma_AB)

    resta_AB = A.resta(B)
    print("\nResta A - B:\n", resta_AB)

    try:
        producto_AB = A.multiplicacion(B)
        print("\nMultiplicación A * B:\n", producto_AB)
    except ValueError as e:
        print("Error en multiplicación:", e)
