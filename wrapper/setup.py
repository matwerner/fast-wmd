from distutils.core import setup    
from distutils.extension import Extension    
from Cython.Build import cythonize    
from Cython.Distutils import build_ext

extensions = cythonize(
	Extension("fastwmd", 
			  sources=["fastwmd.pyx"],
			  language="c++",
			  libraries=["ortools"],
			  extra_compile_args=["-std=c++11", "-O3"]),
	compiler_directives={'language_level' : "3"}
	)

setup(
	name = 'fastwmd',
	cmdclass = {'build_ext': build_ext},
	packages=[],
	ext_modules = extensions
)