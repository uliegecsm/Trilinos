find_package(cudss REQUIRED)

if(cudss_FOUND)
    tribits_extpkg_create_imported_all_libs_target_and_config_file(
        CUDSS
		INNER_FIND_PACKAGE_NAME cudss
        IMPORTED_TARGETS_FOR_ALL_LIBS cudss)
endif()
