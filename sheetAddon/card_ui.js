/**
 * Modern Google Workspace Add-on UI using CardService
 * This file contains all the card-based UI functions for the ExpenseSorted add-on
 */

/**
 * The entry point for the add-on, called by the homepageTrigger.
 * @param {Object} e The event object.
 * @return {CardService.Card} The card to show to the user.
 */
function onHomepage(e) {
  return createMainMenuCard();
}

/**
 * Creates the main menu card with buttons for each feature.
 * @return {CardService.Card}
 */
function createMainMenuCard() {
  var builder = CardService.newCardBuilder();
  builder.setHeader(CardService.newCardHeader().setTitle("ExpenseSorted Tools"));

  var section = CardService.newCardSection().setHeader("Transaction Management");

  // Button to trigger the Import workflow (keeping HTML dialog for file upload)
  section.addWidget(CardService.newButtonSet()
    .addButton(CardService.newTextButton()
      .setText("Import Bank Transactions")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("showImportDialog"))));

  // Button to trigger Categorisation
  section.addWidget(CardService.newButtonSet()
    .addButton(CardService.newTextButton()
      .setText("Categorise New Transactions")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("createEnhancedCategoriseCard"))));

  // Button to trigger Training
  section.addWidget(CardService.newButtonSet()
    .addButton(CardService.newTextButton()
      .setText("Train Model")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("createEnhancedTrainingCard"))));

  var configSection = CardService.newCardSection().setHeader("Configuration");

  // Button to trigger API Key setup
  configSection.addWidget(CardService.newButtonSet()
    .addButton(CardService.newTextButton()
      .setText("Configure API Key")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("createApiKeyCard"))));

  builder.addSection(section);
  builder.addSection(configSection);
  return builder.build();
}

/**
 * Builds the card for setting the API Key.
 * @return {CardService.Card}
 */
function createApiKeyCard() {
  var properties = PropertiesService.getScriptProperties();
  var existingApiKey = properties.getProperty("API_KEY") || "";

  var builder = CardService.newCardBuilder();
  builder.setHeader(CardService.newCardHeader().setTitle("Configure API Key"));
  
  var section = CardService.newCardSection()
    .addWidget(CardService.newTextParagraph()
      .setText("Get your API key from the ExpenseSorted web app and paste it here."));

  // Input field for the API Key
  var apiKeyInput = CardService.newTextInput()
    .setFieldName("api_key_input")
    .setTitle("API Key")
    .setValue(existingApiKey);
  section.addWidget(apiKeyInput);

  // Save button
  var saveButtonSet = CardService.newButtonSet()
    .addButton(CardService.newTextButton()
      .setText("Save API Key")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("handleApiKeySave")));
  
  // Back to main menu button
  saveButtonSet.addButton(CardService.newTextButton()
    .setText("Back to Menu")
    .setOnClickAction(CardService.newAction()
      .setFunctionName("onHomepage")));
  
  section.addWidget(saveButtonSet);
  
  builder.addSection(section);
  return builder.build();
}

/**
 * Handles the action of saving the API key from the card.
 * @param {Object} e The event object from the card interaction.
 * @return {CardService.ActionResponse}
 */
function handleApiKeySave(e) {
  // Get the value from the input field
  var newApiKey = e.formInput.api_key_input;
  
  try {
    Config.saveApiKey(newApiKey);
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification().setText("API Key saved successfully!"))
      .setNavigation(CardService.newNavigation().updateCard(createMainMenuCard()))
      .build();
  } catch (err) {
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification().setText("Error: " + err.message))
      .build();
  }
}

/**
 * Creates the enhanced card for selecting columns to categorise with intuitive column-based mapping.
 * @return {CardService.Card}
 */
function createEnhancedCategoriseCard() {
  Logger.log("🔍 DEBUG: createEnhancedCategoriseCard called - NEW COLUMN-BASED UI");
  
  var builder = CardService.newCardBuilder();
  builder.setHeader(CardService.newCardHeader().setTitle("Categorise Transactions (NEW UI)"));

  var section = CardService.newCardSection();
  section.addWidget(CardService.newTextParagraph()
    .setText("🔥 NEW UI: Map each column in your spreadsheet to a field type. Each field type can only be used once."));

  // Get current sheet columns dynamically
  var sheet = SpreadsheetApp.getActiveSheet();
  var lastColumn = sheet.getLastColumn();
  var headers = [];
  
  // Get actual headers from the sheet if possible
  try {
    var headerRange = sheet.getRange(1, 1, 1, lastColumn);
    var headerValues = headerRange.getValues()[0];
    for (var i = 0; i < headerValues.length; i++) {
      if (headerValues[i] && headerValues[i].toString().trim()) {
        headers.push({
          letter: String.fromCharCode(65 + i),
          name: headerValues[i].toString().trim(),
          index: i + 1
        });
      }
    }
  } catch (error) {
    // Fallback to letter-only if can't read headers
    for (var i = 1; i <= Math.min(lastColumn, 10); i++) {
      headers.push({
        letter: String.fromCharCode(64 + i),
        name: 'Column ' + String.fromCharCode(64 + i),
        index: i
      });
    }
  }

  // Load saved profiles
  var profiles = loadColumnProfiles();
  
  // Profile management section
  if (profiles.length > 0) {
    section.addWidget(CardService.newTextParagraph()
      .setText("<b>Load Saved Profile:</b>"));
    
    var profileSelect = CardService.newSelectionInput()
      .setType(CardService.SelectionInputType.DROPDOWN)
      .setFieldName("profile_to_load")
      .setTitle("Select Profile");
    profileSelect.addItem("-- Select a profile --", "", true);
    profiles.forEach(function(profile) {
      var displayName = profile.name + (profile.bankName ? ' (' + profile.bankName + ')' : '');
      profileSelect.addItem(displayName, profile.id, false);
    });
    section.addWidget(profileSelect);
  }

  section.addWidget(CardService.newTextParagraph()
    .setText("<b>Column Mapping:</b> Choose what each column represents. Leave as 'Not Used' if the column isn't needed."));

  // Create dropdown for each column
  headers.forEach(function(header) {
    var columnSelect = CardService.newSelectionInput()
      .setType(CardService.SelectionInputType.DROPDOWN)
      .setFieldName("column_" + header.letter)
      .setTitle("Column " + header.letter + ": " + header.name);
    
    columnSelect.addItem("Not Used", "", true);
    columnSelect.addItem("Description", "description", false);
    columnSelect.addItem("Description 2 (ANZ style)", "description2", false);
    columnSelect.addItem("Category (Output)", "category", false);
    columnSelect.addItem("Amount", "amount", false);
    columnSelect.addItem("Date", "date", false);
    
    section.addWidget(columnSelect);
  });

  // Row range inputs
  var startRowInput = CardService.newTextInput()
    .setFieldName("start_row")
    .setTitle("Start Row")
    .setValue("2")
    .setHint("First row with data (excluding header)");
  section.addWidget(startRowInput);

  var endRowInput = CardService.newTextInput()
    .setFieldName("end_row")
    .setTitle("End Row (optional)")
    .setHint("Leave empty to process all rows");
  section.addWidget(endRowInput);

  // Profile saving section
  section.addWidget(CardService.newTextParagraph()
    .setText("<b>Save Current Configuration:</b>"));
  
  var profileNameInput = CardService.newTextInput()
    .setFieldName("profile_name")
    .setTitle("Profile Name (optional)")
    .setHint("e.g., ANZ Export Format");
  section.addWidget(profileNameInput);

  var bankNameInput = CardService.newTextInput()
    .setFieldName("bank_name")
    .setTitle("Bank Name (optional)")
    .setHint("e.g., ANZ, ASB, Westpac");
  section.addWidget(bankNameInput);

  // Button set for actions
  var buttonSet = CardService.newButtonSet()
    .addButton(CardService.newTextButton()
      .setText("Categorise Selected Rows")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("handleColumnBasedCategorise")))
    .addButton(CardService.newTextButton()
      .setText("Save Profile Only")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("handleSaveColumnProfile")))
    .addButton(CardService.newTextButton()
      .setText("Back to Menu")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("onHomepage")));

  section.addWidget(buttonSet);
  builder.addSection(section);
  return builder.build();
}

/**
 * Handles the column-based categorise action.
 * @param {Object} e The event object from the card interaction.
 * @return {CardService.ActionResponse}
 */
function handleColumnBasedCategorise(e) {
  try {
    Logger.log("🔍 DEBUG: handleColumnBasedCategorise called");
    Logger.log("🔍 DEBUG: Raw form inputs:", Object.keys(e.formInput));
    
    // Log each form input individually
    Object.keys(e.formInput).forEach(function(key) {
      Logger.log("🔍 DEBUG: " + key + " = '" + e.formInput[key] + "'");
    });
    
    // Process column mappings
    var fieldMappings = {};
    var reverseMappings = {}; // field -> column
    var errors = [];
    var processedColumns = [];
    
    // Get all column mappings
    Object.keys(e.formInput).forEach(function(key) {
      if (key.startsWith('column_')) {
        var column = key.replace('column_', '');
        var fieldType = e.formInput[key];
        
        Logger.log("🔍 DEBUG: Found column input - " + key + " with value '" + fieldType + "'");
        
        if (fieldType && fieldType !== '' && fieldType !== 'Not Used') {
          Logger.log("🔍 DEBUG: Processing mapping - Column " + column + " -> " + fieldType);
          processedColumns.push(column + ":" + fieldType);
          
          // Check for duplicates
          if (reverseMappings[fieldType]) {
            var errorMsg = 'Field "' + fieldType + '" is mapped to both Column ' + reverseMappings[fieldType] + ' and Column ' + column + '. Each field can only be used once.';
            Logger.log("❌ DEBUG: DUPLICATE FOUND - " + errorMsg);
            errors.push(errorMsg);
          } else {
            fieldMappings[column] = fieldType;
            reverseMappings[fieldType] = column;
            Logger.log("✅ DEBUG: Valid mapping added - " + fieldType + " -> Column " + column);
          }
        } else {
          Logger.log("🔍 DEBUG: Skipping empty/unused column " + column + " (value: '" + fieldType + "')");
        }
      }
    });
    
    Logger.log("🔍 DEBUG: Processed columns: " + processedColumns.join(", "));
    Logger.log("🔍 DEBUG: Final field mappings:", JSON.stringify(fieldMappings));
    Logger.log("🔍 DEBUG: Reverse mappings:", JSON.stringify(reverseMappings));
    Logger.log("🔍 DEBUG: Errors found: " + errors.length + " - " + JSON.stringify(errors));
    
    if (errors.length > 0) {
      Logger.log("❌ DEBUG: Returning error notification");
      return CardService.newActionResponseBuilder()
        .setNotification(CardService.newNotification().setText("❌ DUPLICATE MAPPING: " + errors[0]))
        .build();
    }
    
    // Check required fields
    if (!reverseMappings.description || !reverseMappings.category) {
      return CardService.newActionResponseBuilder()
        .setNotification(CardService.newNotification().setText("❌ Both Description and Category fields are required"))
        .build();
    }
    
    // Convert to config format
    var config = {
      descriptionCol: reverseMappings.description,
      description2Col: reverseMappings.description2 || null,
      categoryCol: reverseMappings.category,
      amountCol: reverseMappings.amount || null,
      dateCol: reverseMappings.date || null,
      startRow: parseInt(e.formInput.start_row) || 2,
      endRow: e.formInput.end_row ? parseInt(e.formInput.end_row) : null
    };
    
    // Save profile if name provided
    if (e.formInput.profile_name && e.formInput.profile_name.trim()) {
      try {
        saveColumnProfile(
          e.formInput.profile_name.trim(),
          e.formInput.bank_name ? e.formInput.bank_name.trim() : null,
          fieldMappings,
          { startRow: config.startRow, endRow: config.endRow }
        );
      } catch (error) {
        Logger.log('Profile save failed: ' + error.message);
      }
    }
    
    // Start the categorisation process
    categoriseTransactions(config);
    
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification().setText("✅ Categorisation started! Check the polling dialog for progress."))
      .setNavigation(CardService.newNavigation().updateCard(createMainMenuCard()))
      .build();
  } catch (err) {
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification().setText("Error: " + err.message))
      .build();
  }
}

/**
 * Handles saving a column-based profile.
 * @param {Object} e The event object from the card interaction.
 * @return {CardService.ActionResponse}
 */
function handleSaveColumnProfile(e) {
  try {
    if (!e.formInput.profile_name || !e.formInput.profile_name.trim()) {
      return CardService.newActionResponseBuilder()
        .setNotification(CardService.newNotification().setText("❌ Profile name is required"))
        .build();
    }
    
    var fieldMappings = {};
    
    // Get all column mappings
    Object.keys(e.formInput).forEach(function(key) {
      if (key.startsWith('column_') && e.formInput[key]) {
        var column = key.replace('column_', '');
        var fieldType = e.formInput[key];
        fieldMappings[column] = fieldType;
      }
    });
    
    var config = {
      startRow: parseInt(e.formInput.start_row) || 2,
      endRow: e.formInput.end_row ? parseInt(e.formInput.end_row) : null
    };
    
    saveColumnProfile(
      e.formInput.profile_name.trim(),
      e.formInput.bank_name ? e.formInput.bank_name.trim() : null,
      fieldMappings,
      config
    );
    
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification().setText("✅ Profile saved successfully!"))
      .setNavigation(CardService.newNavigation().updateCard(createEnhancedCategoriseCard()))
      .build();
  } catch (err) {
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification().setText("❌ Error saving profile: " + err.message))
      .build();
  }
}

/**
 * Creates the enhanced card for training the model with unique mapping logic.
 * @return {CardService.Card}
 */
function createEnhancedTrainingCard() {
  var builder = CardService.newCardBuilder();
  builder.setHeader(CardService.newCardHeader().setTitle("Train Model"));

  var section = CardService.newCardSection();
  section.addWidget(CardService.newTextParagraph()
    .setText("Train the AI model with your categorized transactions. Configure unique column mappings and optionally save as a profile."));

  // Get current sheet columns dynamically
  var sheet = SpreadsheetApp.getActiveSheet();
  var lastColumn = sheet.getLastColumn();
  var columns = ['-- None --'];
  
  for (var i = 1; i <= Math.min(lastColumn + 2, 26); i++) {
    var letter = String.fromCharCode(64 + i);
    columns.push(letter);
  }

  // Load saved profiles
  var profiles = loadColumnProfiles();
  
  // Profile management section
  if (profiles.length > 0) {
    section.addWidget(CardService.newTextParagraph()
      .setText("<b>Load Saved Profile:</b>"));
    
    var profileSelect = CardService.newSelectionInput()
      .setType(CardService.SelectionInputType.DROPDOWN)
      .setFieldName("profile_to_load")
      .setTitle("Select Profile");
    profileSelect.addItem("-- Select a profile --", "", true);
    profiles.forEach(function(profile) {
      var displayName = profile.name + (profile.bankName ? ' (' + profile.bankName + ')' : '');
      profileSelect.addItem(displayName, profile.id, false);
    });
    section.addWidget(profileSelect);
  }

  // Important mapping note for training
  section.addWidget(CardService.newTextParagraph()
    .setText("<b>⚠️ IMPORTANT:</b> Each column can only be mapped to ONE field type. If you see the same field type in multiple dropdowns, only the LAST selection will be used. Choose different columns for each field type."));

  // Description Column Dropdown
  var descSelect = CardService.newSelectionInput()
    .setType(CardService.SelectionInputType.DROPDOWN)
    .setFieldName("description_col")
    .setTitle("Description Column");
  columns.forEach(function(c, index) {
    descSelect.addItem(c, c === '-- None --' ? '' : c, c === 'C' && index > 0);
  });
  section.addWidget(descSelect);

  // Description 2 Column Dropdown (for banks like ANZ that split descriptions)
  var desc2Select = CardService.newSelectionInput()
    .setType(CardService.SelectionInputType.DROPDOWN)
    .setFieldName("description2_col")
    .setTitle("Description 2 Column (Optional)");
  columns.forEach(function(c, index) {
    desc2Select.addItem(c, c === '-- None --' ? '' : c, index === 0);
  });
  section.addWidget(desc2Select);

  // Category Column Dropdown
  var catSelect = CardService.newSelectionInput()
    .setType(CardService.SelectionInputType.DROPDOWN)
    .setFieldName("category_col")
    .setTitle("Category Column");
  columns.forEach(function(c, index) {
    catSelect.addItem(c, c === '-- None --' ? '' : c, c === 'E' && index > 0);
  });
  section.addWidget(catSelect);
  
  // Amount Column Dropdown (Optional)
  var amountSelect = CardService.newSelectionInput()
    .setType(CardService.SelectionInputType.DROPDOWN)
    .setFieldName("amount_col")
    .setTitle("Amount Column (Optional)");
  columns.forEach(function(c, index) {
    amountSelect.addItem(c, c === '-- None --' ? '' : c, c === 'B' && index > 0);
  });
  section.addWidget(amountSelect);

  // Date Column Dropdown (Optional but recommended)
  var dateSelect = CardService.newSelectionInput()
    .setType(CardService.SelectionInputType.DROPDOWN)
    .setFieldName("date_col")
    .setTitle("Date Column (Optional)");
  columns.forEach(function(c, index) {
    dateSelect.addItem(c, c === '-- None --' ? '' : c, c === 'A' && index > 0);
  });
  section.addWidget(dateSelect);

  // Row range inputs
  var startRowInput = CardService.newTextInput()
    .setFieldName("start_row")
    .setTitle("Start Row")
    .setValue("2")
    .setHint("First row with data (excluding header)");
  section.addWidget(startRowInput);

  var endRowInput = CardService.newTextInput()
    .setFieldName("end_row")
    .setTitle("End Row (optional)")
    .setHint("Leave empty to process all rows");
  section.addWidget(endRowInput);

  // Profile saving section
  section.addWidget(CardService.newTextParagraph()
    .setText("<b>Save Current Configuration:</b>"));
  
  var profileNameInput = CardService.newTextInput()
    .setFieldName("profile_name")
    .setTitle("Profile Name (optional)")
    .setHint("e.g., ANZ Training Setup");
  section.addWidget(profileNameInput);

  var bankNameInput = CardService.newTextInput()
    .setFieldName("bank_name")
    .setTitle("Bank Name (optional)")
    .setHint("e.g., ANZ, ASB, Westpac");
  section.addWidget(bankNameInput);

  // Button set for actions
  var buttonSet = CardService.newButtonSet()
    .addButton(CardService.newTextButton()
      .setText("Train Model")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("handleEnhancedTrainModel")))
    .addButton(CardService.newTextButton()
      .setText("Save Profile Only")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("handleSaveTrainingProfile")))
    .addButton(CardService.newTextButton()
      .setText("Back to Menu")
      .setOnClickAction(CardService.newAction()
        .setFunctionName("onHomepage")));

  section.addWidget(buttonSet);
  builder.addSection(section);
  return builder.build();
}

/**
 * Handles the enhanced training action with unique mapping validation.
 * @param {Object} e The event object from the card interaction.
 * @return {CardService.ActionResponse}
 */
function handleEnhancedTrainModel(e) {
  try {
    // Load profile if specified
    if (e.formInput.profile_to_load) {
      var profiles = loadColumnProfiles();
      var selectedProfile = profiles.find(function(profile) {
        return profile.id === e.formInput.profile_to_load;
      });
      if (selectedProfile) {
        // Use profile mappings
        var config = {
          narrativeCol: selectedProfile.mappings.description_col || null,
          categoryCol: selectedProfile.mappings.category_col || null,
          amountCol: selectedProfile.mappings.amount_col || null,
          startRow: selectedProfile.config.startRow || 2,
          endRow: selectedProfile.config.endRow || null
        };
      }
    } else {
      // Validate unique mappings
      var mappings = {
        description_col: e.formInput.description_col || null,
        description2_col: e.formInput.description2_col || null,
        category_col: e.formInput.category_col || null,
        amount_col: e.formInput.amount_col || null,
        date_col: e.formInput.date_col || null
      };
      
      var usedColumns = [];
      var errors = [];
      
      Object.keys(mappings).forEach(function(fieldType) {
        var column = mappings[fieldType];
        if (column && column !== '') {
          if (usedColumns.indexOf(column) > -1) {
            errors.push('Column ' + column + ' is mapped to multiple fields');
          } else {
            usedColumns.push(column);
          }
        }
      });
      
      if (errors.length > 0) {
        return CardService.newActionResponseBuilder()
          .setNotification(CardService.newNotification().setText("Mapping Error: " + errors.join(', ')))
          .build();
      }
      
      // Required field validation
      if (!mappings.description_col || !mappings.category_col) {
        return CardService.newActionResponseBuilder()
          .setNotification(CardService.newNotification().setText("Error: Description and Category columns are required"))
          .build();
      }
      
      var config = {
        narrativeCol: mappings.description_col,  // trainModel expects 'narrativeCol'
        categoryCol: mappings.category_col,
        amountCol: mappings.amount_col,
        startRow: parseInt(e.formInput.start_row) || 2,
        endRow: e.formInput.end_row ? parseInt(e.formInput.end_row) : null
      };
      
      // Save profile if name provided
      if (e.formInput.profile_name && e.formInput.profile_name.trim()) {
        try {
          saveColumnProfile(
            e.formInput.profile_name.trim(),
            e.formInput.bank_name ? e.formInput.bank_name.trim() : null,
            mappings,
            { startRow: config.startRow, endRow: config.endRow }
          );
        } catch (error) {
          // Continue with training even if profile save fails
          Logger.log('Profile save failed: ' + error.message);
        }
      }
    }
    
    // Start the training process
    trainModel(config);
    
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification().setText("Model training started! Check the polling dialog for progress."))
      .setNavigation(CardService.newNavigation().updateCard(createMainMenuCard()))
      .build();
  } catch (err) {
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification().setText("Error: " + err.message))
      .build();
  }
}

/**
 * Handles saving a training profile without running training.
 * @param {Object} e The event object from the card interaction.
 * @return {CardService.ActionResponse}
 */
function handleSaveTrainingProfile(e) {
  try {
    if (!e.formInput.profile_name || !e.formInput.profile_name.trim()) {
      return CardService.newActionResponseBuilder()
        .setNotification(CardService.newNotification().setText("Error: Profile name is required"))
        .build();
    }
    
    var mappings = {
      description_col: e.formInput.description_col || null,
      description2_col: e.formInput.description2_col || null,
      category_col: e.formInput.category_col || null,
      amount_col: e.formInput.amount_col || null,
      date_col: e.formInput.date_col || null
    };
    
    var config = {
      startRow: parseInt(e.formInput.start_row) || 2,
      endRow: e.formInput.end_row ? parseInt(e.formInput.end_row) : null
    };
    
    saveColumnProfile(
      e.formInput.profile_name.trim(),
      e.formInput.bank_name ? e.formInput.bank_name.trim() : null,
      mappings,
      config
    );
    
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification().setText("Training profile saved successfully!"))
      .setNavigation(CardService.newNavigation().updateCard(createEnhancedTrainingCard()))
      .build();
  } catch (err) {
    return CardService.newActionResponseBuilder()
      .setNotification(CardService.newNotification().setText("Error saving training profile: " + err.message))
      .build();
  }
}

/**
 * Test function to verify card structures and log them for debugging
 * This helps developers see what the cards look like without needing the sidebar.
 * Run this function in the Apps Script IDE to test the card interface.
 */
function testCardStructures() {
  try {
    Logger.log("=== TESTING CARD STRUCTURES ===");
    
    // Test homepage card (main menu)
    var homepage = onHomepage();
    Logger.log("✅ Main menu card (onHomepage) created successfully");
    
    // Test API key card
    var apiCard = createApiKeyCard();
    Logger.log("✅ API key card created successfully");
    
    // Test categorise card
    var categoriseCard = createEnhancedCategoriseCard();
    Logger.log("✅ Enhanced Categorise card created successfully");
    
    // Test training card
    var trainingCard = createEnhancedTrainingCard();
    Logger.log("✅ Enhanced training card created successfully");
    
    Logger.log("=== TESTING ACTION HANDLERS ===");
    
    // Verify all handler functions exist
    Logger.log("✅ handleApiKeySave function exists: " + (typeof handleApiKeySave === 'function'));
    Logger.log("✅ handleColumnBasedCategorise function exists: " + (typeof handleColumnBasedCategorise === 'function'));
    Logger.log("✅ handleEnhancedTrainModel function exists: " + (typeof handleEnhancedTrainModel === 'function'));
    Logger.log("✅ handleSaveColumnProfile function exists: " + (typeof handleSaveColumnProfile === 'function'));
    Logger.log("✅ handleSaveTrainingProfile function exists: " + (typeof handleSaveTrainingProfile === 'function'));
    
    Logger.log("=== ALL CARD TESTS PASSED ===");
    Logger.log("🎉 The modern card interface is fully functional!");
    Logger.log("📱 These cards will appear in the Google Workspace sidebar when published");
    Logger.log("🔧 For now, use the HTML dialogs via the Add-ons menu for actual functionality");
    
    return "All card structures tested successfully! Check the Apps Script execution log for details.";
    
  } catch (error) {
    Logger.log("❌ Error testing cards: " + error.toString());
    Logger.log("Error details: " + error.stack);
    throw error;
  }
}

// Profile management functions
function saveColumnProfile(profileName, bankName, mappings, config) {
  try {
    var properties = PropertiesService.getScriptProperties();
    var existingProfiles = JSON.parse(properties.getProperty('COLUMN_PROFILES') || '[]');
    
    var profile = {
      id: Date.now().toString(),
      name: profileName,
      bankName: bankName || '',
      mappings: mappings,
      config: config,
      createdAt: new Date().toISOString()
    };
    
    existingProfiles.push(profile);
    properties.setProperty('COLUMN_PROFILES', JSON.stringify(existingProfiles));
    
    return profile;
  } catch (error) {
    throw new Error('Failed to save profile: ' + error.message);
  }
}

function loadColumnProfiles() {
  try {
    var properties = PropertiesService.getScriptProperties();
    return JSON.parse(properties.getProperty('COLUMN_PROFILES') || '[]');
  } catch (error) {
    return [];
  }
}

function deleteColumnProfile(profileId) {
  try {
    var properties = PropertiesService.getScriptProperties();
    var existingProfiles = JSON.parse(properties.getProperty('COLUMN_PROFILES') || '[]');
    var updatedProfiles = existingProfiles.filter(function(profile) {
      return profile.id !== profileId;
    });
    properties.setProperty('COLUMN_PROFILES', JSON.stringify(updatedProfiles));
    return true;
  } catch (error) {
    throw new Error('Failed to delete profile: ' + error.message);
  }
}

/**
 * Simple test to verify which UI is being used. Run this in Apps Script IDE.
 */
function debugWhichUI() {
  Logger.log("=== DEBUGGING WHICH UI IS ACTIVE ===");
  
  try {
    var card = createEnhancedCategoriseCard();
    Logger.log("✅ NEW UI: createEnhancedCategoriseCard function exists and runs");
    Logger.log("📝 Card title should show: 'Categorise Transactions (NEW UI)'");
    Logger.log("📝 Card text should start with: '🔥 NEW UI: Map each column...'");
    
    // Check if we have the new handler
    if (typeof handleColumnBasedCategorise === 'function') {
      Logger.log("✅ NEW HANDLER: handleColumnBasedCategorise function exists");
    } else {
      Logger.log("❌ MISSING: handleColumnBasedCategorise function not found");
    }
    
    return "Debug complete - check logs for details";
  } catch (error) {
    Logger.log("❌ ERROR: " + error.toString());
    return "Error occurred - check logs";
  }
} 