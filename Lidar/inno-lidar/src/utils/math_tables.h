/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef UTILS_MATH_TABLES_H_
#define UTILS_MATH_TABLES_H_

#include <stddef.h>
#include <stdint.h>

#include <math.h>

#include "utils/log.h"

// #define EXACT_SIN_COS
#define EXACT_ASIN_ATAN

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace innovusion {

/**
 * @brief MathTables
 */
class MathTables {
 private:
  // must be same as defined in sdk/inno_lidar_packet.h
  static const size_t kInnoAngleUnitPerPiRad = 32768;
  static constexpr double kRadPerInnoAngleUnit = M_PI / kInnoAngleUnitPerPiRad;

  static const int32_t kDegreeInPi = 180;
  static constexpr double kInnoAngleUnitPerOneDegree =
    kInnoAngleUnitPerPiRad / static_cast<double>(kDegreeInPi);
  static const int32_t kAngleTableSize = 2 * kInnoAngleUnitPerPiRad;

  static const int32_t kASinTableSize = 10000;
  static const int32_t kATanTableSize = 45000;
  static const int32_t kAtanTableScale = 10000;
  static const int32_t kAsinTableScale = 10000;

 private:
#ifndef EXACT_SIN_COS
  static float sin_table_[];
  static float cos_table_[];
#endif

#ifndef EXACT_ASIN_ATAN
  static int32_t asin_table_[];
  static int32_t atan_table_[];
#endif

 public:
  /**
   * @brief Verify the unit is kInnoAngleUnitPerPiRad
   * @param unit Unit to be verified
   */
  inline static void verify_unit(size_t unit) {
    inno_log_verify(unit == kInnoAngleUnitPerPiRad,
                    "unit doesn't match %" PRI_SIZELU
                    " vs %" PRI_SIZELU,
                    unit, kInnoAngleUnitPerPiRad);
  }
  /**
   * @brief Convert degree to radian
   * @param v Degree to be converted
   * @return Return the radian
   */
  inline static double degree_to_rad(double v) {
    return v * M_PI / kDegreeInPi;
  }

  /**
   * @brief Convert degree to inno angle unit.
   *        The inno angle unit is defined as kInnoAngleUnitPerPiRad per pi radian.
   *        The inno angle unit is defined as kInnoAngleUnitPerOneDegree per degree.
   * @param theta Degree to be converted
   * @return Return the inno angle unit
   */
  inline static double get_scaled_theta(double theta) {
    if (theta < 0) {
      int32_t multiple = 1 + (static_cast<int>(theta) / -360);
      theta += 360 * multiple;
    }
    if (theta >= 360) {
      int32_t multiple = static_cast<int32_t>(theta) / 360;
      theta -= 360 * multiple;
    }
    return theta * kInnoAngleUnitPerOneDegree;
  }

  /**
   * @brief Exactly caclulate sin value in inno angle unit
   * @param i Inno angle unit
   * @return Return the sin value
   */
  inline static double lookup_sin_table_in_unit_exact(int i) {
    return sin(i * kRadPerInnoAngleUnit);
  }

  /**
   * @brief Exactly calculate sin value or lookup sin table in inno angle unit
   *        If EXACT_SIN_COS is defined, then calculate sin value exactly.
   *        Otherwise, lookup sin table.
   * @param i Inno angle unit
   * @return Return the sin value
   */
  inline static double lookup_sin_table_in_unit(int i) {
#ifdef EXACT_SIN_COS
    return lookup_sin_table_in_unit_exact(i);
#else
    return sin_table_[i];
#endif
  }

  /**
   * @brief Exactly calculate cos value in inno angle unit
   * @param i Inno angle unit
   * @return Return the cos value
   */
  inline static double lookup_cos_table_in_unit_exact(int i) {
    return cos(i * kRadPerInnoAngleUnit);
  }

  /**
   * @brief Exactly calculate cos value or lookup cos table in inno angle unit
   *        If EXACT_SIN_COS is defined, then calculate cos value exactly.
   *        Otherwise, lookup cos table.
   * @param i Inno angle unit
   * @return Return the cos value
   */
  inline static double lookup_cos_table_in_unit(int i) {
#ifdef EXACT_SIN_COS
    return lookup_cos_table_in_unit_exact(i);
#else
    return cos_table_[i];
#endif
  }

  /**
   * @brief Exactly lookup asin table in degree
   * @param theta Degree
   * @return Return the sin value
   */
  inline static double lookup_sin_table_exact(double theta) {
    return sin(degree_to_rad(theta));
  }

  /**
   * @brief Exactly calculate sin value or lookup sin table in degree
   *        If EXACT_SIN_COS is defined, then calculate sin value exactly.
   *        Otherwise, lookup sin table.
   * @param theta Degree
   * @return Return the sin value
   */
  inline static double lookup_sin_table(double theta) {
#ifdef EXACT_SIN_COS
    return lookup_sin_table_exact(theta);
#else
    double scaled_theta;
    int32_t truncated_theta;

    scaled_theta = get_scaled_theta(theta);
    truncated_theta = static_cast<int32_t>(scaled_theta);
#if 0
    if (truncated_theta < 0 || truncated_theta >= kStepsInOneDegree * 360) {
      inno_log_error("bad angle %f %d", theta, truncated_theta);
      truncated_theta = 0;
    }
#endif
    return sin_table_[truncated_theta] + (scaled_theta - truncated_theta) *
    (sin_table_[truncated_theta + 1] - sin_table_[truncated_theta]);
#endif
  }

  /**
   * @brief Exactly calculate cos value in degree
   * @param theta Degree
   * @return Return the cos value
   */
  inline static double lookup_cos_table_exact(double theta) {
    return cos(degree_to_rad(theta));
  }

  /**
   * @brief Exactly calculate cos value or lookup cos table in degree
   *        If EXACT_SIN_COS is defined, then calculate cos value exactly.
   *        Otherwise, lookup cos table.
   * @param theta Degree
   * @return Return the cos value
   */
  inline static double lookup_cos_table(double theta) {
#ifdef EXACT_SIN_COS
    return lookup_cos_table_exact(theta);
#else
    double scaled_theta;
    int32_t truncated_theta;

    scaled_theta = get_scaled_theta(theta);
    truncated_theta = static_cast<int32_t>(scaled_theta);
#if 0
    if (truncated_theta < 0 || truncated_theta >= kStepsInOneDegree * 360) {
      inno_log_error(stderr, "bad angle %f %d", theta, truncated_theta);
      truncated_theta = 0;
    }
#endif
    return cos_table_[truncated_theta] + (scaled_theta - truncated_theta) *
        (cos_table_[truncated_theta + 1] - cos_table_[truncated_theta]);
#endif
  }

  /**
   * @brief Exactly calculate atan value
   * @param value  The value to be calculated
   * @param result The result of atan value in inno angle unit
   * @return Return 0 if success, otherwise return -1
   */
  inline static int lookup_atan_table_exact(double value, int32_t *result) {
    *result = static_cast<int>(atan(value) / kRadPerInnoAngleUnit);
    return 0;
  }

  inline static int lookup_atan2_table_exact(double y, double x, int32_t *result) {
    *result = static_cast<int>(atan2(y, x) / kRadPerInnoAngleUnit);
    return 0;
  }

  /**
   * @brief Exactly calculate atan value or lookup atan table
   *        If EXACT_ASIN_ATAN is defined, then calculate atan value exactly.
   *        Otherwise, lookup atan table.
   * @param value  The value to be calculated
   * @param result The result of atan value in inno angle unit
   * @return Return 0 if success, otherwise return -1
   */
  inline static int lookup_atan_table(double value, int32_t *result) {
#ifdef EXACT_ASIN_ATAN
    return lookup_atan_table_exact(value, result);
#else
    int atan_table_index = static_cast<int>(value * kAtanTableScale +
                                            kATanTableSize);
    if (atan_table_index < 0 || atan_table_index >= 2 * kATanTableSize) {
      inno_log_error("atan_table_index: %d out of bound", atan_table_index);
      return -1;
    } else {
      *result = atan_table_[atan_table_index];
      return 0;
    }
#endif
  }

  /**
   * @brief Exactly calculate asin value
   * @param value  The value to be calculated
   * @param result The result of asin value in inno angle unit
   * @return Return 0 if success, otherwise return -1
   */
  inline static int lookup_asin_table_exact(double value, int32_t *result) {
    *result = static_cast<int>(asin(value) / kRadPerInnoAngleUnit);
    return 0;
  }
  /**
   * @brief Exactly calculate asin value or lookup asin table
   * @param value  The value to be calculated
   * @param result The result of asin value in inno angle unit
   * @return Return 0 if success, otherwise return -1
   */
  inline static int lookup_asin_table(double value, int32_t *result) {
#ifdef EXACT_ASIN_ATAN
    return lookup_asin_table_exact(value, result);
#else
    int asin_table_index = static_cast<int>(value * kAsinTableScale +
                                            kASinTableSize);
    if (asin_table_index < 0 || asin_table_index >= 2 * kASinTableSize) {
      inno_log_error("asin_table_index: %d out of bound", asin_table_index);
      return -1;
    } else {
      *result = asin_table_[asin_table_index];
      return 0;
    }
#endif
  }

 public:
  /**
   * @brief TablesInit class
   */
  static class TablesInit {
   public:
    /**
     * @brief TablesInit constructor
     *        Setup sin, cos, atan and asin table
     * @param inno_angle_unit radian per inno angle unit
     */
    TablesInit(double inno_angle_unit) {
      setup_table_(inno_angle_unit);
    }

   private:
    /**
     * @brief Setup sin, cos, atan and asin table
     * @param inno_angle_unit radian per inno angle unit
     */
    void setup_table_(double inno_angle_unit);
  } tables_init;
};

}  // namespace innovusion

#endif  // UTILS_MATH_TABLES_H_
