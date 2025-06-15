from setuptools import setup, find_packages

setup(
    name='opt_trans',                            # Choose a unique name for your package
    version='1.0.0',                        # Start with a small version number
    packages=find_packages(where='src'),    # This automatically finds packages in the src directory
    package_dir={'': 'src'},                # Tells setuptools that packages are under src directory
)