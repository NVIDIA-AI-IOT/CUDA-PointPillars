/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#ifndef UTILS_CONFIG_H_
#define UTILS_CONFIG_H_

#include <limits.h>

#include <list>
#include <mutex>  // NOLINT
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>

#include "utils/log.h"

namespace innovusion {

/**
 * @brief Config
 */
class Config {
 public:
  Config();
  virtual ~Config();
  /**
   * @brief Check if the config type is the same as the input
   * @param n Config type to be checked
   * @return Return true if the config type is the same as the input
   */
  bool is_same_type(const std::string &n);
  /**
   * @brief Set the key value of the config
   * @param cfg_key_in   Key of the config
   * @param cfg_value_in Value of the config
   * @return Return 0 if success, others for error
   */
  int set_key_value(const std::string &cfg_key_in,
                    const std::string &cfg_value_in);
  /**
   * @brief Get the config type in string
   * @return Return the config type in string
   */
  virtual const char* get_type() const = 0;
  /**
   * @brief Copy the config from the input, only copy when version are different
   * @param src Config to be copied
   * @return Return ture for copied, false for not copied
   */
  bool copy_from_src(Config *src);

 protected:
  /**
   * Set a double value config
   * @param key   Config key name
   * @param value Config value in double
   * @return Return 0 if success, others for error
   */
  virtual int set_key_value_(const std::string &key,
                             double value) = 0;
  /**
   * set a string value config
   * @param key   Config key name
   * @param value Config value in string
   * @return Return 0 if success, others for error
   */
  virtual int set_key_value_(const std::string &key,
                             const std::string value) {
    // do nothing in base config
    return -1;
  }
  /**
   * @brief Get the start address of the config
   * @return Return the start address of the config
   */
  virtual void *get_start_() = 0;
  /**
   * @brief Get the size of the config
   * @return Return the size of the config
   */
  virtual size_t get_size_() = 0;

  /**
   * @brief Get the version of the config
   * @return Return the version of the config
   */
  inline uint64_t get_version_locked_() {
    // inno_log_assert(mutex_.owns_lock(), "not locked");
    return version_;
  }

  /**
   * @brief Set the version of the config
   * @param r Version to be set
   * @return Return the version of the config
   */
  inline uint64_t set_version_locked_(uint64_t r) {
    // inno_log_assert(mutex_.owns_lock(), "not locked");
    version_ = r;
    return r;
  }

  /**
   * @brief Increase the version of the config
   */
  void inc_version_();

 protected:
  std::mutex mutex_;

 private:
  uint64_t version_;
};

#define BEGIN_CFG_MEMBER()                      \
 private:                                       \
  void *get_start_() override {                  \
    return &start_;                             \
  }                                             \
  size_t get_size_() override {                           \
    return uintptr_t(&end_) - uintptr_t(get_start_());   \
  }                                                      \
  int32_t start_;                                        \
  public:

#define END_CFG_MEMBER()                        \
 private:                                       \
  int32_t end_;

#define SET_CFG(name)                           \
  do {                                          \
     if (strcmp(key.c_str(), #name) == 0) {     \
       std::unique_lock<std::mutex> lk(mutex_); \
       (name) = value;                            \
       return 0;                                \
     }                                          \
  } while (0)

/**
 * @brief ConfigManager
 */
class ConfigManager {
 public:
  /**
   * @brief ConfigManager constructor
   * @param basename Base name of the config.
   *                 Configs managed by this ConfigManager should be named starting with this basename.
   */
  explicit ConfigManager(const char *basename);
  ~ConfigManager();
  /**
   * @brief Add a config to the ConfigManager.
   *        Config's name should start with the basename of the ConfigManager.
   * @param c Config to be added
   */
  void add_config(Config *c);
  /**
   * @brief Remove config from the ConfigManager.
   * @param c Config to be removed.
   *          All configs with the same name and address will be removed.
   */
  void remove_config(Config *c);
  /**
   * @brief Read configs from history and set to the configs list.
   */
  void play_config();
  /**
   * @brief Set the key value of the config
   * @param cfg_key_in   Key of the config
   * @param cfg_value_in Value of the config
   * @param from_app     True if the config is set from app, false if from server
   * @return Return 0 if success, others for error
   */
  int set_config_key_value(const std::string &cfg_key_in,
                           const std::string &cfg_value_in,
                           bool from_app);

 protected:
  std::mutex mutex_;

 private:
  const char *basename_;
  std::unordered_map<std::string, std::vector<Config *>> configs_;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::string>> history_;
};

/***********************
  Example Config class
 ***********************/
class ExampleConfig: public Config {
 public:
  ExampleConfig() : Config() {
    test1 = 0;  // <== ADD_MEMBER_STEP1: init the member
    test2 = 0;
    test3 = 0;
  }
  const char* get_type() const override {
    return "Example";  // <== different Config class must return different name
  }

  int set_key_value_(const std::string &key,
                             double value) override {
    SET_CFG(test1);  // <== ADD_MEMBER_STEP2
    SET_CFG(test2);
    SET_CFG(test3);
    return -1;
  }

  int set_key_value_(const std::string &key,
                     const std::string value) override {
    // no string attribute
    return -1;
  }

  BEGIN_CFG_MEMBER()
  uint64_t test1;  // <== ADD_MEMBER_STEP3: declare type
  bool test2;
  double test3;
  END_CFG_MEMBER()
};


typedef std::function<void(const std::string &, const std::string &)>
                        GlobalEnvListener;

/**
 * @brief GlobalEnv
 */
class GlobalEnv {
 private:
  GlobalEnv(void) {}

 public:
  /**
   * @brief Get the instance of GlobalEnv
   * @return Return the instance of GlobalEnv
   */
  static GlobalEnv * getInstance() {
    static GlobalEnv  cfg;
    return &cfg;
  }

  /**
   * @brief Get the value of the environment variable
   * @param name Name of the environment variable
   * @param val  Value of the environment variable
   * @return Return true if the environment variable exists, false if not
   */
  bool get(const std::string & name, std::string & val) { // NOLINT
    std::unique_lock<std::mutex> lk(mutex_);
    auto itr = env_.find(name);
    if (itr == env_.end()) {
      return false;
    }
    val = itr->second;
    return true;
  }

  /**
   * @brief Set the value of the environment variable
   * @param name  Name of the environment variable
   * @param value Value of the environment variable
   */
  inline void set(const std::string & name, int value) {
    set(name, std::to_string(value));
  }

  /**
   * @brief Set the value of the environment variable.
   *        And then call each listener of the environment variable.
   * @param name  Name of the environment variable
   * @param value Value of the environment variable
   */
  void set(const std::string & name, const std::string & value) {
    std::unique_lock<std::mutex> lk(mutex_);
    auto env_itr = env_.find(name);
    if (env_itr == env_.end() || value != env_itr->second) {
      env_[name] = value;
      for (auto & p : env_listeners_[name]) {
        p.second(name, value);
      }
    }
  }

  /**
   * @brief Set environment listener
   * @param listener_name Name of the listener
   * @param name          Name of the environment variable
   * @param listener      Listener function
   */
  void listen(const std::string & listener_name, const std::string & name,
          const GlobalEnvListener & listener) {
    std::unique_lock<std::mutex> lk(mutex_);
    auto itr = env_.find(name);
    if (itr != env_.end()) {
      listener(name, itr->second);
    }
    env_listeners_[name][listener_name] = listener;
  }

  /**
   * @brief Remove environment listener
   * @param listener_name Name of the listener
   * @param name          Name of the environment variable
   */
  void unlisten(const std::string & listener_name, const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex_);
    auto itr_listeners = env_listeners_.find(name);
    if (itr_listeners != env_listeners_.end()) {
      itr_listeners->second.erase(listener_name);
    }
  }

  /**
   * @brief Remove all listeners of a given environment variable
   * @param name Name of the environment variable
   */
  inline void clear(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex_);
    env_.erase(name);
    env_listeners_.erase(name);
  }

  /**
   * @brief Remove all environment variables listeners
   */
  inline void clear(void) {
    std::unique_lock<std::mutex> lk(mutex_);
    env_.clear();
    env_listeners_.clear();
  }

 private:
  std::unordered_map<std::string, std::string> env_;
  std::unordered_map<std::string,
      std::unordered_map<std::string, GlobalEnvListener>> env_listeners_;
  std::mutex mutex_;
};

}  // namespace innovusion
#endif  // UTILS_CONFIG_H_
