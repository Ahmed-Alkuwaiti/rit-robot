from setuptools import find_packages, setup

package_name = 'rit_robocomp_2026'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/templates', [
            'templates/template_1.png',
            'templates/template_2.png',
            'templates/template_3.png',
            'templates/template_4.png',
            'templates/template_5.png',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/bringup.launch.py',
            'launch/solution.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='RIT RoboComp 2026 perception + navigation',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'perception_node = rit_robocomp_2026.perception_node:main',
            'navigation_node = rit_robocomp_2026.navigation_node:main',
            'manipulation_node = rit_robocomp_2026.manipulation_node:main',
            'task_manager = rit_robocomp_2026.task_manager:main',
        ],
    },
)
