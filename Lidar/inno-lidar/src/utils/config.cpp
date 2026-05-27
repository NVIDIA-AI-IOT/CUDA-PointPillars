/**
 *  Copyright (C) 2021 - Innovusion Inc.
 *
 *  All Rights Reserved.
 *
 *  $Id$
 */

#include "utils/config.h"

#include <limits.h>

#include <algorithm>
#include <string>
#include <unordered_map>

#include "utils/log.h"
#include "utils/utils.h"

namespace innovusion {

Config::Config() {
  version_ = 0;
}

Config::~Config() {
}

bool Config::is_same_type(const std::string &n) {
  return (strcmp(get_type(), n.c_str()) == 0);
}

int Config::set_key_value(const std::string &cfg_key_in,
                          const std::string &cfg_value_in) {
  std::string key(cfg_key_in);
  std::string value(cfg_value_in);
  InnoUtils::trim_space(&key);
  InnoUtils::trim_space(&value);
  // value is enclosed in double quotation marks, regard as a string
  int r = -1;
  if (!cfg_value_in.empty()) {
    if (value.at(0) == '"' && value.at(value.size() - 1) == '"') {
      r = set_key_value_(key, *InnoUtils::trim(&value, "\""));
    } else {
      double v;
      if (sscanf(value.c_str(), "%lf", &v) != 1) {
        inno_log_error("%s invalid config value %s (key=%s)", get_type(), cfg_value_in.c_str(), cfg_key_in.c_str());
        return -1;
      }
      r = set_key_value_(key, v);
    }
  }
  if (r == 0) {
    inc_version_();
    inno_log_info("config %s(%" PRI_SIZEU ") set %s to %s",
                  get_type(),
                  version_,
                  key.c_str(),
                  value.c_str());
  } else {
    inno_log_error("config %s invalid key %s (value=%s)",
                   get_type(),
                   cfg_key_in.c_str(),
                   cfg_value_in.c_str());
  }
  return r;
}

bool Config::copy_from_src(Config *src) {
  /* have to be from the same class */
  inno_log_verify(strcmp(src->get_type(), get_type()) == 0,
                  "invalid copy %s to %s",
                  src->get_type(),
                  get_type());
  bool copied = false;
  {
    std::unique_lock<std::mutex> lk(src->mutex_);
    {
      std::unique_lock<std::mutex> lk2(mutex_);
          if (src->get_version_locked_() != get_version_locked_()) {
            memcpy(get_start_(),
                   src->get_start_(),
                   src->get_size_());
            set_version_locked_(src->get_version_locked_());
            copied = true;
          }
    }
  }
  return copied;
}

void Config::inc_version_() {
  std::unique_lock<std::mutex> lk(mutex_);
  version_++;
  return;
}

ConfigManager::ConfigManager(const char *basename)
    : basename_(basename)
    , configs_() {
}

ConfigManager::~ConfigManager() {
}

void ConfigManager::play_config() {
  std::unique_lock<std::mutex> lk(mutex_);
  for (auto& p : history_) {
    const std::string& section = p.first;
    std::unordered_map<std::string, std::string>& a = p.second;
    for (Config* config : configs_[section]) {
      for (auto& nv_pair : a) {
        config->set_key_value(nv_pair.first, nv_pair.second);
      }
    }
  }
}

void ConfigManager::add_config(Config *c) {
  inno_log_verify(c, "NULL config point");

  std::string name(c->get_type());
  InnoUtils::trim_space(&name);

  // must start with basename_
  inno_log_verify(name.find(basename_) == 0,
                  "%s config module name must start with %s",
                  name.c_str(), basename_);
  {
    std::unique_lock<std::mutex> lk(mutex_);
    std::unordered_map<std::string, std::vector<Config*>>::iterator itconf =
      configs_.find(name);
    /*
      inno_log_verify(it == configs_.end(),
                     "type conflict, same type %s already in manager",
                     name.c_str());
    */
    if (itconf == configs_.end()) {
      configs_[name] = std::vector<Config *>(0);
    }
    configs_[name].push_back(c);

    // replay history
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::string>>::iterator i
      = history_.find(name);
    if (i != history_.end()) {
      std::unordered_map<std::string, std::string> &a = history_[name];
      for (std::unordered_map<std::string, std::string>::iterator it =
               a.begin();
           it != a.end(); ++it ) {
        const std::string& hname = it->first;
        const std::string& hvalue = it->second;
        c->set_key_value(hname, hvalue);
      }
    }
  }
  return;
}

void ConfigManager::remove_config(Config *c) {
  inno_log_verify(c, "NULL config point");
  std::string name(c->get_type());
  InnoUtils::trim_space(&name);
  inno_log_verify(c != NULL, "%s config is NULL", name.c_str());

  {
    std::unique_lock<std::mutex> lk(mutex_);
        std::unordered_map<std::string, std::vector<Config*>>::iterator it =
        configs_.find(name);
    inno_log_verify(it != configs_.end(),
                    "cannot find config %s",
                    name.c_str());
    std::vector<Config*> &vec = it->second;
    vec.erase(std::remove(vec.begin(), vec.end(), c), vec.end());
  }
  return;
}

int ConfigManager::set_config_key_value(const std::string &cfg_name_in,
                                        const std::string &cfg_value_in,
                                        bool from_app) {
  std::string whole_name(cfg_name_in);
  InnoUtils::trim_space(&whole_name);

  size_t epos = whole_name.find("/");
  if (epos == std::string::npos || epos == 0||
      epos >= whole_name.size() - 1) {
    inno_log_warning("bad config %s=%s",
                     cfg_name_in.c_str(), cfg_value_in.c_str());
    return -1;
  }
  std::string section = whole_name.substr(0, epos);
  std::string key = whole_name.substr(epos + 1);
  int r = 0;

  {
    std::unique_lock<std::mutex> lk(mutex_);
    if (from_app) {
      std::unordered_map<std::string,
                     std::unordered_map<std::string, std::string>>::iterator i
        = history_.find(section);
      if (i == history_.end()) {
        std::unordered_map<std::string, std::string> a;
        history_[section] = a;
      }
      history_[section][key] = cfg_value_in;
    }

    std::unordered_map<std::string, std::vector<Config *>>::iterator it =
        configs_.find(section);
    if (it == configs_.end()) {
      inno_log_info("%s config value %s (key=%s), will be applied later",
                    section.c_str(),
                    key.c_str(),
                    cfg_value_in.c_str());
      // WYY to: seperate config from load
      return -2;
    }

    std::vector<Config *> &vec = it->second;
    for (size_t i = 0; i < vec.size(); i++) {
      int k = vec[i]->set_key_value(key,
                                    cfg_value_in);
      if (k != 0) {
        r = k;
      }
    }
  }
  return r;
}

}  // namespace innovusion
