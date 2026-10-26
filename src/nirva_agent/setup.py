from setuptools import find_packages, setup

package_name = 'nirva_agent'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'openai', 'opencv-python'],
    zip_safe=True,
    maintainer='Megha',
    maintainer_email='you@example.com',
    description='NIRVA: Nemotron-powered closed-loop Physical AI agent for TurtleBot3 + Nav2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'agent_node = nirva_agent.agent_node:main',
            'perception_node = nirva_agent.perception_node:main',
        ],
    },
)