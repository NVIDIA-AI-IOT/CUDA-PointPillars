/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "utils/math_tables.h"

namespace innovusion {
#ifndef EXACT_SIN_COS
float MathTables::sin_table_[kAngleTableSize + 1];
float MathTables::cos_table_[kAngleTableSize + 1];
#endif

#ifndef EXACT_ASIN_ATAN
int32_t MathTables::asin_table_[2 * kASinTableSize];
int32_t MathTables::atan_table_[2 * kATanTableSize];
#endif

MathTables::TablesInit MathTables::tables_init(
    MathTables::kRadPerInnoAngleUnit);

void MathTables::TablesInit::setup_table_(double inno_angle_unit) {
#ifndef EXACT_SIN_COS
  for (int32_t i = 0; i <= kAngleTableSize; ++i) {
    double angle = i * kRadPerInnoAngleUnit;
    cos_table_[i] = cos(angle);
    sin_table_[i] = sin(angle);
  }
#endif

#ifndef EXACT_ASIN_ATAN
  for (int32_t i = -kASinTableSize; i < kASinTableSize; i++) {
    asin_table_[i + kASinTableSize] =
        static_cast<int32_t>(asin(i/static_cast<double>(kAsinTableScale))
                             / inno_angle_unit);
  }

  for (int32_t i = -kATanTableSize; i < kATanTableSize; i++) {
    atan_table_[i + kATanTableSize] =
        static_cast<int32_t>(atan(i/static_cast<double>(kAtanTableScale))
                             / inno_angle_unit);
  }
#endif
}

}  // namespace innovusion
