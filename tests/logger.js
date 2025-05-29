// logger.js - Simple logging utilities
const LOG_LEVELS = {
  ERROR: 0,
  INFO: 1, 
  DEBUG: 2,
  TRACE: 3,
};

let CURRENT_LOG_LEVEL = LOG_LEVELS.INFO;

const log = (message, level = LOG_LEVELS.INFO) => {
  if (level <= CURRENT_LOG_LEVEL) {
    const timestamp = new Date().toISOString();
    const prefix =
      level === LOG_LEVELS.ERROR
        ? "[ERROR] "
        : level === LOG_LEVELS.DEBUG
        ? "[DEBUG] "
        : level === LOG_LEVELS.TRACE
        ? "[TRACE] "
        : "";
    console.log(`[${timestamp}] ${prefix}${message}`);
  }
};

const logError = (message) => log(message, LOG_LEVELS.ERROR);
const logInfo = (message) => log(message, LOG_LEVELS.INFO);
const logDebug = (message) => log(message, LOG_LEVELS.DEBUG);
const logTrace = (message) => log(message, LOG_LEVELS.TRACE);

module.exports = {
  LOG_LEVELS,
  logError,
  logInfo,
  logDebug,
  logTrace,
  setLogLevel: (level) => { CURRENT_LOG_LEVEL = level; }
};
