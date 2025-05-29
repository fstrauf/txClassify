// sheetAddon/config.js
const CLASSIFICATION_SERVICE_URL = "https://api.expensesorted.com";

var Config = {
  /**
   * Retrieves the service configuration, including URL and API key.
   * @return {object} The service configuration.
   * @throws {Error} If the API key is not configured.
   */
  getServiceConfig: function () {
    var properties = PropertiesService.getScriptProperties();
    var apiKey = properties.getProperty("API_KEY");

    if (!apiKey) {
      throw new Error(
        'API key not configured. Please go to expensesorted.com to get your API key, then use "Configure API Key" to set it up.'
      );
    }

    return {
      serviceUrl: CLASSIFICATION_SERVICE_URL,
      apiKey: apiKey,
      // userId: Session.getEffectiveUser().getEmail() // Removed from here
    };
  },

  /**
   * Saves the API key to script properties.
   * @param {string} apiKey The API key to save.
   * @throws {Error} If the API key is not provided.
   */
  saveApiKey: function (apiKey) {
    if (!apiKey || typeof apiKey !== "string" || apiKey.trim() === "") {
      throw new Error("API key is required and cannot be empty.");
    }

    var properties = PropertiesService.getScriptProperties();
    properties.setProperty("API_KEY", apiKey.trim());
    // Logging/status update will be handled by the calling function in main.js
  },
};
