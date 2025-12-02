# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2022  Jan Brezovsky, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta <janbre@amu.edu.pl>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import find_packages, setup

install_requires = ['numpy>=1.17.3',
                    'scipy>=1.5.2',
                    'scikit-learn>=0.24.0',
                    'biopython>=1.78',
                    'fastcluster>=1.1.25',
                    'hdbscan>=0.8.24',
                    'mdtraj',
                    'threadpoolctl',
                    'joblib<1.2.0'
                    ]

with open('./README.md', 'r') as fh:
    file_content = fh.read()

setup(name='transport_tools',
      version=__import__('transport_tools').__version__,
      description='a library for massive analyses of internal voids in biomolecules and ligand transport through them',
      long_description=file_content,
      long_description_content_type="text/markdown",
      url='http://labbit.eu/software',
      project_urls={
          'Documentation': 'https://github.com/labbit-eu/transport_tools',
          'Source': 'https://github.com/labbit-eu/transport_tools',
          'Tracker': 'https://github.com/labbit-eu/transport_tools/issues',
      },
      author=__import__('transport_tools').__author__,
      author_email=__import__('transport_tools').__mail__,
      license='GNU GPL v3',
      keywords='molecular-dynamics transport tunnels channels',
      packages=find_packages(include=['transport_tools*']),
      python_requires='>=3.8',
      install_requires=install_requires,
      scripts=['transport_tools/scripts/tt_engine.py','transport_tools/scripts/tt_filter_caver_by_frames.py','transport_tools/scripts/tt_convert_to_caver.py'],
      provides=['transport_tools'],
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'Operating System :: POSIX',
                   'Programming Language :: Python',
                   ],
      include_package_data=True
      )
