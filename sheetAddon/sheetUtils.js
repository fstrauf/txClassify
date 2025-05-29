var SheetUtils = {
  /**
   * Updates the status in the Log sheet.
   * @param {string} message The main message for the log.
   * @param {string} [additionalDetails=""] Optional additional details.
   */
  updateStatus: function (message, additionalDetails = "") {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var logSheet = ss.getSheetByName("Log");

    // Create Log sheet if it doesn't exist
    if (!logSheet) {
      logSheet = ss.insertSheet("Log");
      logSheet.getRange("A1:D1").setValues([["Timestamp", "Status", "Message", "Details"]]);
      logSheet.setFrozenRows(1);
      logSheet.setColumnWidth(1, 180);
      logSheet.setColumnWidth(2, 100);
      logSheet.setColumnWidth(3, 300);
      logSheet.setColumnWidth(4, 400);
      var headerRange = logSheet.getRange("A1:D1");
      headerRange.setBackground("#f3f3f3").setFontWeight("bold").setHorizontalAlignment("center");
    }

    var timestamp = new Date().toLocaleString();
    var status = "INFO";
    if (message.toLowerCase().includes("error")) {
      status = "ERROR";
    } else if (message.toLowerCase().includes("completed") || message.toLowerCase().includes("success")) {
      status = "SUCCESS";
    } else if (message.toLowerCase().includes("progress") || message.toLowerCase().includes("processing")) {
      status = "PROCESSING";
    }

    var activeSheet = SpreadsheetApp.getActiveSheet().getName();
    var contextDetails = additionalDetails || `Active Sheet: ${activeSheet}`;
    logSheet.insertRowAfter(1);
    logSheet.getRange("A2:D2").setValues([[timestamp, status, message, contextDetails]]);

    var statusCell = logSheet.getRange("B2");
    switch (status) {
      case "ERROR":
        statusCell.setBackground("#ffcdd2");
        break;
      case "SUCCESS":
        statusCell.setBackground("#c8e6c9");
        break;
      case "PROCESSING":
        statusCell.setBackground("#fff9c4");
        break;
      default:
        statusCell.setBackground("#ffffff");
    }
    logSheet.getRange("A2:D2").setHorizontalAlignment("left").setVerticalAlignment("middle").setWrap(true);
    var lastRow = logSheet.getLastRow();
    if (lastRow > 101) {
      logSheet.deleteRows(102, lastRow - 101);
    }
    logSheet.autoResizeColumns(1, 4);
    if (logSheet.isSheetHidden()) {
      logSheet.showSheet();
    }
  },

  /**
   * Gets or creates the Settings sheet.
   * @return {GoogleAppsScript.Spreadsheet.Sheet} The Settings sheet.
   */
  getSettingsSheet: function () {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var settingsSheet = ss.getSheetByName("Settings");
    if (!settingsSheet) {
      settingsSheet = ss.insertSheet("Settings");
      settingsSheet.getRange("A1:B1").setValues([["Setting", "Value"]]);
      settingsSheet.setFrozenRows(1);
      settingsSheet.setColumnWidth(1, 200);
      settingsSheet.setColumnWidth(2, 300);
      settingsSheet.hideSheet();
    }
    return settingsSheet;
  },

  /**
   * Updates a specific setting in the Settings sheet.
   * @param {string} settingName The name of the setting.
   * @param {any} value The value of the setting.
   */
  updateSetting: function (settingName, value) {
    var sheet = this.getSettingsSheet(); // Use 'this' to call other methods in the object
    var data = sheet.getDataRange().getValues();
    var rowIndex = -1;
    for (var i = 1; i < data.length; i++) {
      if (data[i][0] === settingName) {
        rowIndex = i + 1;
        break;
      }
    }
    if (rowIndex === -1) {
      rowIndex = data.length + 1;
    }
    sheet.getRange(rowIndex, 1, 1, 2).setValues([[settingName, value]]);
  },

  /**
   * Gets or creates the Stats sheet.
   * @return {GoogleAppsScript.Spreadsheet.Sheet} The Stats sheet.
   */
  getStatsSheet: function () {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var statsSheet = ss.getSheetByName("Stats");
    if (!statsSheet) {
      statsSheet = ss.insertSheet("Stats");
      statsSheet.getRange("A1:B1").setValues([["Metric", "Value"]]);
      statsSheet.setFrozenRows(1);
      statsSheet.setColumnWidth(1, 200);
      statsSheet.setColumnWidth(2, 300);
      var headerRange = statsSheet.getRange("A1:B1");
      headerRange.setBackground("#f3f3f3").setFontWeight("bold").setHorizontalAlignment("center");
      statsSheet
        .getRange("A2:A5")
        .setValues([["Last Training Time"], ["Training Data Size"], ["Training Sheet"], ["Model Status"]]);
    }
    return statsSheet;
  },

  /**
   * Updates a specific metric in the Stats sheet.
   * @param {string} metric The name of the metric.
   * @param {any} value The value of the metric.
   */
  updateStats: function (metric, value) {
    var sheet = this.getStatsSheet(); // Use 'this' to call other methods in the object
    var data = sheet.getDataRange().getValues();
    var rowIndex = -1;
    for (var i = 1; i < data.length; i++) {
      if (data[i][0] === metric) {
        rowIndex = i + 1;
        break;
      }
    }
    if (rowIndex === -1) {
      rowIndex = data.length + 1;
      sheet.getRange(rowIndex, 1).setValue(metric);
    }
    sheet.getRange(rowIndex, 2).setValue(value);
  },

  /**
   * Writes classification results to the specified sheet.
   * @param {object} result The API response containing classification results.
   * @param {object} config Configuration object with column and row details.
   * @param {GoogleAppsScript.Spreadsheet.Sheet} sheet The sheet to write results to.
   * @return {boolean} True if results were written successfully, false otherwise.
   */
  writeResultsToSheet: function (result, config, sheet) {
    try {
      Logger.log("Writing results to sheet with config: " + JSON.stringify(config));
      if (!sheet || typeof sheet.getRange !== "function") {
        Logger.log("Invalid sheet object. Attempting to get active sheet.");
        sheet = SpreadsheetApp.getActiveSheet();
        if (!sheet) {
          throw new Error("Could not obtain a valid sheet to write results.");
        }
      }

      var resultsData = this.findResultsArray(result);

      if (!resultsData || resultsData.length === 0) {
        Logger.log("No results array found. Full result: " + JSON.stringify(result));
        this.updateStatus("Processing complete, but no results found.");
        return false;
      }

      Logger.log("Found " + resultsData.length + " results to write");
      var categoryCol = config.categoryCol;
      var confidenceCol = config.confidenceCol;
      var moneyInOutCol = config.moneyInOutCol;
      var startRow = parseInt(config.startRow);
      var endRow = parseInt(config.endRow);

      if (!categoryCol || !startRow || !endRow || startRow > endRow) {
        throw new Error(`Invalid range or category column (Rows ${startRow}-${endRow}, Col ${categoryCol}).`);
      }

      if (endRow - startRow + 1 !== resultsData.length) {
        Logger.log(
          `Result count (${resultsData.length}) mismatch with range size (${endRow - startRow + 1}). Adjusting.`
        );
        endRow = startRow + resultsData.length - 1;
        this.updateStatus(
          `Warning: Result count mismatch. Results written from row ${startRow}.`,
          `Sheet: ${sheet.getName()}`
        );
      }

      var categories = resultsData.map((r) => [r.predicted_category || r.category || r.Category || ""]);
      sheet.getRange(categoryCol + startRow + ":" + categoryCol + endRow).setValues(categories);

      var hasConfidence = resultsData.some(
        (r) => r.hasOwnProperty("similarity_score") || r.hasOwnProperty("confidence") || r.hasOwnProperty("score")
      );
      if (confidenceCol && hasConfidence) {
        var confidenceScores = resultsData.map((r) => {
          var score =
            r.similarity_score !== undefined
              ? r.similarity_score
              : r.confidence !== undefined
              ? r.confidence
              : r.score !== undefined
              ? r.score
              : "";
          return [score];
        });
        try {
          var confidenceRange = sheet.getRange(confidenceCol + startRow + ":" + confidenceCol + endRow);
          confidenceRange.setValues(confidenceScores);
          if (confidenceScores.some((s) => typeof s[0] === "number" && s[0] >= 0 && s[0] <= 1)) {
            confidenceRange.setNumberFormat("0.00%");
          }
        } catch (e) {
          this.updateStatus(`Warning: Could not write confidence scores to col ${confidenceCol}.`, `Error: ${e}`);
        }
      }

      var hasMoneyInFlag = resultsData.some((r) => r.hasOwnProperty("money_in"));
      if (moneyInOutCol && hasMoneyInFlag) {
        var moneyInValues = resultsData.map((r) => {
          if (r.money_in === true) return ["IN"];
          if (r.money_in === false) return ["OUT"];
          return [""];
        });
        try {
          sheet.getRange(moneyInOutCol + startRow + ":" + moneyInOutCol + endRow).setValues(moneyInValues);
        } catch (e) {
          this.updateStatus(`Warning: Could not write Money In/Out flags to col ${moneyInOutCol}.`, `Error: ${e}`);
        }
      }

      var statusMsg = `Categorised ${resultsData.length} transactions successfully!`;
      this.updateStatus(statusMsg, `Sheet: '${sheet.getName()}', Rows: ${startRow}-${endRow}`);

      try {
        const statsSheet = this.getStatsSheet();
        const metrics = statsSheet.getRange("A2:A").getValues().flat();
        let currentValue = 0;
        for (let i = 0; i < metrics.length; i++) {
          if (metrics[i] === "categorisations") {
            const val = statsSheet.getRange(`B${i + 2}`).getValue();
            if (typeof val === "number") currentValue = val;
            break;
          }
        }
        this.updateStats("categorisations", currentValue + resultsData.length);
      } catch (statError) {
        Logger.log("Could not update categorisation stats: " + statError);
      }

      return true;
    } catch (error) {
      Logger.log("Error writing results: " + error.toString());
      SpreadsheetApp.getUi().alert("Error writing results: " + error.toString());
      return false;
    }
  },

  /**
   * Finds the array of results within a potentially nested API response object.
   * @param {object} result The API response.
   * @return {Array|null} The array of result items, or null if not found.
   */
  findResultsArray: function (result) {
    if (result.results && Array.isArray(result.results)) return result.results;
    if (result.data && Array.isArray(result.data)) return result.data;
    if (result.result && typeof result.result === "object") {
      if (result.result.results && Array.isArray(result.result.results)) return result.result.results;
      if (result.result.data && Array.isArray(result.result.data)) return result.result.data;
    }
    for (var key in result) {
      if (
        result.hasOwnProperty(key) &&
        Array.isArray(result[key]) &&
        result[key].length > 0 &&
        (result[key][0].hasOwnProperty("predicted_category") ||
          result[key][0].hasOwnProperty("Category") ||
          result[key][0].hasOwnProperty("category"))
      ) {
        return result[key];
      }
    }
    if (
      Array.isArray(result) &&
      result.length > 0 &&
      (result[0].hasOwnProperty("predicted_category") ||
        result[0].hasOwnProperty("Category") ||
        result[0].hasOwnProperty("category"))
    ) {
      return result;
    }
    return null;
  },

  /**
   * Converts a column number (1-indexed) to its letter representation (e.g., 1 -> A, 27 -> AA).
   * @param {number} column The column number.
   * @return {string} The column letter(s).
   */
  columnToLetter: function (column) {
    var temp,
      letter = "";
    while (column > 0) {
      temp = (column - 1) % 26;
      letter = String.fromCharCode(temp + 65) + letter;
      column = (column - temp - 1) / 26;
    }
    return letter;
  },
};
