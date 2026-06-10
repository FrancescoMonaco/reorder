#pragma once
#include <mutex>
#include <string>
#include <iostream>
#include <chrono>

namespace club {

enum LogLevel {
  ERROR,
  WARNING,
  INFO,
  DEBUG,
  TRACE
};

static LogLevel CURRENT_LEVEL = LogLevel::INFO;

static void set_log_level(LogLevel level) {
  CURRENT_LEVEL = level;
}

template<typename T>
static void log_format(T value) {
  std::cout << value;
}


template<>
void log_format(std::string value) {
  std::cout << "\"" << value << "\"";
}
template<>
void log_format(const char * value) {
  std::cout << "\"" << value << "\"";
}
template<>
void log_format(LogLevel value) {
  switch (value) {
    case LogLevel::ERROR:   std::cout << "ERROR"; break;
    case LogLevel::WARNING: std::cout << "WARN"; break;
    case LogLevel::INFO:    std::cout << "INFO"; break;
    case LogLevel::DEBUG:   std::cout << "DEBUG"; break;
    case LogLevel::TRACE:   std::cout << "TRACE"; break;
  }
}

static void do_log() {
  std::cout << std::endl;
}

template<typename V, typename... Others>
static void do_log(const char * k, V v, Others... others) {
  std::cout << " " << k << "=";
  log_format(v);
  do_log(others...);
}

static std::mutex mutx;

template<typename V, typename... Others>
static void log(LogLevel level, const char * k, V v, Others... others) {
  if (level > CURRENT_LEVEL) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutx);
  std::chrono::duration since_epoch = std::chrono::system_clock::now().time_since_epoch();
  auto time = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
  do_log("level", level, "time", time.count(), k, v, others...);
}

#define LOG_ERROR(args...) log(club::LogLevel::ERROR, args)
#define LOG_WARN(args...)  log(club::LogLevel::WARNING, args)
#define LOG_INFO(args...)  log(club::LogLevel::INFO, args)
#define LOG_DEBUG(args...) log(club::LogLevel::DEBUG, args)
#define LOG_TRACE(args...) log(club::LogLevel::TRACE, args)

};