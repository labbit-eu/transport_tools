#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2021  Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej
# Surpeta, Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal  <janbre@amu.edu.pl>
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

__version__ = '0.9.0'
__author__ = 'Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta, ' \
             'Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal'
__mail__ = 'janbre@amu.edu.pl'

from transport_tools.libs.tools import TransportProcesses, save_checkpoint, load_checkpoint
from transport_tools.libs.ui import license_printer, init_parser
from transport_tools.libs.config import AnalysisConfig
from logging import getLogger
from os.path import join, exists

logger = getLogger(__name__)


def test_checkpoint(checkpoint_file: str):
    """
    Test that we can save checkpoint before running expensive calculations
    """

    if exists(checkpoint_file):
        print("Error checkpoint file '{}' already exists.\nSpecify different name or enable overwrite by using "
              "'--overwrite' option\nor setting 'overwrite' parameter to True\n".format(checkpoint_file))
        exit(0)


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    if args.write_config_file:
        AnalysisConfig().write_template_file("./tmp_config.ini", advanced=args.advanced)
        exit(0)

    if args.print_version:
        print("TransportTools library version {}\n".format(__version__))
        exit(0)

    if args.print_license:
        license_printer()
        exit(0)

    if args.config_filename is None:
        parser.print_help()
        exit(0)

    configuration = AnalysisConfig(args.config_filename)
    start_from_stage = configuration.get_parameter("start_from_stage")
    stop_after_stage = configuration.get_parameter("stop_after_stage")
    checkpoints_path = configuration.get_parameter("checkpoints_folder")
    overwrite = args.overwrite or configuration.get_parameter("overwrite")
    mol_system = TransportProcesses(configuration)

    current_stage = 1
    logger.info("-" * 78)
    logger.info("STAGE %d - Preparatory stage for unified analyses", current_stage)
    logger.info("-" * 78)
    if start_from_stage == current_stage:
        if not overwrite:
            test_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage)))
        mol_system.clear_results(overwrite)
        mol_system.compute_transformations()
        save_checkpoint(mol_system, join(checkpoints_path, "stage{:03d}.dump".format(current_stage)),
                        overwrite=overwrite)
    else:
        logger.info("Requested to start from STAGE %d => skipping STAGE %d", start_from_stage, current_stage)

    if current_stage == stop_after_stage:
        logger.info("Requested to stop after this STAGE")
        exit(0)

    current_stage = 2
    logger.info("-" * 78)
    logger.info("STAGE %d - Processing datasets of tunnel networks from CAVER results", current_stage)
    logger.info("-" * 78)
    if start_from_stage == current_stage:
        mol_system = load_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage - 1)),
                                     update_config=configuration)
    if start_from_stage <= current_stage:
        if not overwrite:
            test_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage)))
        mol_system.process_tunnel_networks()
        save_checkpoint(mol_system, join(checkpoints_path, "stage{:03d}.dump".format(current_stage)),
                        overwrite=overwrite)
    else:
        logger.info("Requested to start from STAGE %d => skipping STAGE %d", start_from_stage, current_stage)

    if current_stage == stop_after_stage:
        logger.info("Requested to stop after this STAGE")
        exit(0)

    current_stage = 3
    logger.info("-" * 78)
    logger.info("STAGE %d - Layering tunnel clusters to get their simplified representation", current_stage)
    logger.info("-" * 78)
    if start_from_stage == current_stage:
        mol_system = load_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage - 1)),
                                     update_config=configuration)
    if start_from_stage <= current_stage:
        if not overwrite:
            test_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage)))
        mol_system.create_layered_description4tunnel_networks()
        save_checkpoint(mol_system, join(checkpoints_path, "stage{:03d}.dump".format(current_stage)),
                        overwrite=overwrite)
    else:
        logger.info("Requested to start from STAGE %d => skipping STAGE %d", start_from_stage, current_stage)

    if current_stage == stop_after_stage:
        logger.info("Requested to stop after this STAGE")
        exit(0)

    current_stage = 4
    logger.info("-" * 78)
    logger.info("STAGE %d - Computing distances among the layered clusters", current_stage)
    logger.info("-" * 78)
    if start_from_stage == current_stage:
        mol_system = load_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage - 1)),
                                     update_config=configuration)
    if start_from_stage <= current_stage:
        if not overwrite:
            test_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage)))

        mol_system.compute_tunnel_clusters_distances()
        save_checkpoint(mol_system, join(checkpoints_path, "stage{:03d}.dump".format(current_stage)),
                        overwrite=overwrite)
    else:
        logger.info("Requested to start from STAGE %d => skipping STAGE %d", start_from_stage, current_stage)

    if current_stage == stop_after_stage:
        logger.info("Requested to stop after this STAGE")
        exit(0)

    current_stage = 5
    logger.info("-" * 78)
    logger.info("STAGE %d - Clustering the layered clusters into superclusters and creating initial outputs",
                current_stage)
    logger.info("-" * 78)
    if start_from_stage == current_stage:
        mol_system = load_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage - 1)),
                                     update_config=configuration)
    if start_from_stage <= current_stage:
        if not overwrite:
            test_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage)))
        mol_system.merge_tunnel_clusters2super_clusters()
        mol_system.create_super_cluster_profiles()
        mol_system.generate_super_cluster_summary(out_filename="1-initial_tunnels_statistics.txt")
        mol_system.save_super_clusters_visualization(script_name="1-visualize_initial_tunnels.py")
        save_checkpoint(mol_system, join(checkpoints_path, "stage{:03d}.dump".format(current_stage)),
                        overwrite=overwrite)
    else:
        logger.info("Requested to start from STAGE %d => skipping STAGE %d", start_from_stage, current_stage)

    if current_stage == stop_after_stage:
        logger.info("Requested to stop after this STAGE")
        exit(0)

    current_stage = 6
    logger.info("-" * 78)
    logger.info("STAGE %d - Filtering superclusters and creating filtered outputs", current_stage)
    logger.info("-" * 78)
    if start_from_stage == current_stage:
        mol_system = load_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage - 1)),
                                     update_config=configuration)
    if start_from_stage <= current_stage:
        if not overwrite:
            test_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage)))
        mol_system.filter_super_cluster_profiles(*configuration.get_filters())
        mol_system.generate_super_cluster_summary(out_filename="2-filtered_tunnels_statistics.txt")
        mol_system.save_super_clusters_visualization(script_name="2-visualize_filtered_tunnels.py")
        save_checkpoint(mol_system, join(checkpoints_path, "stage{:03d}.dump".format(current_stage)),
                        overwrite=overwrite)
    else:
        logger.info("Requested to start from STAGE %d => skipping STAGE %d", start_from_stage, current_stage)

    if current_stage == stop_after_stage:
        logger.info("Requested to stop after this STAGE")
        exit(0)

    if configuration.get_parameter("aquaduct_results_path") is not None:
        current_stage = 7
        logger.info("-" * 78)
        logger.info("STAGE %d - Processing datasets of transport events from AQUA-DUCT results", current_stage)
        logger.info("-" * 78)
        if start_from_stage == current_stage:
            mol_system = load_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage - 1)),
                                         update_config=configuration)
        if start_from_stage <= current_stage:
            if not overwrite:
                test_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage)))
            mol_system.process_aquaduct_networks()
            save_checkpoint(mol_system, join(checkpoints_path, "stage{:03d}.dump".format(current_stage)),
                            overwrite=overwrite)
        else:
            logger.info("Requested to start from STAGE %d => skipping STAGE %d", start_from_stage, current_stage)

        if current_stage == stop_after_stage:
            logger.info("Requested to stop after this STAGE")
            exit(0)

        current_stage = 8
        logger.info("-" * 78)
        logger.info("STAGE %d  - Layering transport events to get their simplified representation", current_stage)
        logger.info("-" * 78)
        if start_from_stage == current_stage:
            mol_system = load_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage - 1)),
                                         update_config=configuration)
        if start_from_stage <= current_stage:
            if not overwrite:
                test_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage)))
            mol_system.create_layered_description4aquaduct_networks()
            save_checkpoint(mol_system, join(checkpoints_path, "stage{:03d}.dump".format(current_stage)),
                            overwrite=overwrite)
        else:
            logger.info("Requested to start from STAGE %d => skipping STAGE %d", start_from_stage, current_stage)

        if current_stage == stop_after_stage:
            logger.info("Requested to stop after this STAGE")
            exit(0)

        current_stage = 9
        logger.info("-" * 78)
        logger.info("STAGE %d - Assigning layered events to tunnel networks in superclusters "
                    "and creating initial outputs with events", current_stage)
        logger.info("-" * 78)
        if start_from_stage == current_stage:
            mol_system = load_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage - 1)),
                                         update_config=configuration)
        if start_from_stage <= current_stage:
            if not overwrite:
                test_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage)))
            mol_system.assign_transport_events()
            mol_system.save_super_clusters_visualization(script_name="3-visualize_initial_events.py")
            mol_system.generate_super_cluster_summary(out_filename="3-initial_events_statistics.txt")
            save_checkpoint(mol_system, join(checkpoints_path, "stage{:03d}.dump".format(current_stage)),
                            overwrite=overwrite)
        else:
            logger.info("Requested to start from STAGE %d => skipping STAGE %d", start_from_stage, current_stage)

        if current_stage == stop_after_stage:
            logger.info("Requested to stop after this STAGE")
            exit(0)

        current_stage = 10
        logger.info("-" * 78)
        logger.info("STAGE %d - Filtering supercluster with events and creating filtered outputs with events",
                    current_stage)
        logger.info("-" * 78)
        if start_from_stage == current_stage:
            mol_system = load_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage - 1)),
                                         update_config=configuration)
        if start_from_stage <= current_stage:
            if not overwrite:
                test_checkpoint(join(checkpoints_path, "stage{:03d}.dump".format(current_stage)))
            mol_system.filter_super_cluster_profiles(*configuration.get_filters())
            mol_system.save_super_clusters_visualization(script_name="4-visualize_filtered_events.py")
            mol_system.generate_super_cluster_summary(out_filename="4-filtered_events_statistics.txt")
            save_checkpoint(mol_system, join(checkpoints_path, "stage{:03d}.dump".format(current_stage)),
                            overwrite=overwrite)
        else:
            logger.info("Requested to start from STAGE %d => skipping STAGE %d", start_from_stage, current_stage)
