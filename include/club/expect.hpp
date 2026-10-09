#pragma once

#include <cstdlib>
#include <iostream>

#define expect( condition )                                           \
    do {                                                              \
        if ( !( condition ) ) {                                       \
            std::cerr << "Assertion failed: (" << #condition << ")\n" \
                      << "File: " << __FILE__ << "\n"                 \
                      << "Line: " << __LINE__ << std::endl;           \
            std::abort();                                             \
        }                                                             \
    } while ( 0 )
