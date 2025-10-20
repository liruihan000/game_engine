#!/usr/bin/env node

/**
 * clearCanvas Exempt Functionality Test Script
 * Test ID format: "001", "002", "003" - numeric string format
 */

// Mock clearCanvas core logic
function testClearCanvas(items, exemptList) {
  console.log('\n=== Test Start ===');
  console.log('Original items:', items.map(item => `${item.id}(${item.type})`).join(', '));
  console.log('Exempt list:', exemptList || 'none');
  
  // Core logic - keep avatar_set and exempted items
  const keptItems = items.filter(item => 
    item.type === "avatar_set" || 
    (exemptList && exemptList.includes(item.id))
  );
  
  const removedItems = items.filter(item => !keptItems.includes(item));
  
  console.log('Kept items:', keptItems.map(item => `${item.id}(${item.type})`).join(', '));
  console.log('Removed items:', removedItems.map(item => `${item.id}(${item.type})`).join(', '));
  
  // Statistics
  const avatarCount = keptItems.filter(item => item.type === "avatar_set").length;
  const exemptCount = exemptList ? exemptList.filter(id => 
    items.some(item => item.id === id && item.type !== "avatar_set")
  ).length : 0;
  const removedCount = items.length - keptItems.length;
  
  const result = `Cleared: removed ${removedCount} items, kept ${avatarCount} avatars${exemptCount > 0 ? ` and ${exemptCount} exempt items` : ''}`;
  console.log('Result:', result);
  console.log('=== Test End ===\n');
  
  return { keptItems, removedItems, result };
}

// Test data
const mockItems = [
  { id: "001", type: "text_display", name: "Welcome Message" },
  { id: "002", type: "phase_indicator", name: "Current Phase" }, 
  { id: "003", type: "avatar_set", name: "Player Avatars" },
  { id: "004", type: "timer", name: "Phase Timer" },
  { id: "005", type: "voting_panel", name: "Vote Panel" },
  { id: "006", type: "score_board", name: "Scoreboard" }
];

console.log('🧪 clearCanvas Exempt Functionality Test');
console.log('=====================================');

// Test Case 1: No exemption (only keep avatar)
console.log('\n📋 Test Case 1: No exempt list');
testClearCanvas(mockItems, null);

// Test Case 2: Exempt one component
console.log('\n📋 Test Case 2: Exempt phase_indicator(002)');
testClearCanvas(mockItems, ["002"]);

// Test Case 3: Exempt multiple components
console.log('\n📋 Test Case 3: Exempt timer(004) and scoreboard(006)');
testClearCanvas(mockItems, ["004", "006"]);

// Test Case 4: Exempt non-existent ID
console.log('\n📋 Test Case 4: Exempt non-existent ID(999)');
testClearCanvas(mockItems, ["999"]);

// Test Case 5: Exempt avatar_set (should have no effect since avatar is kept anyway)
console.log('\n📋 Test Case 5: Exempt avatar_set(003)');
testClearCanvas(mockItems, ["003"]);

// Test Case 6: Complex scenario - exempt multiple + include avatar
console.log('\n📋 Test Case 6: Complex scenario - exempt 001,003,005');
testClearCanvas(mockItems, ["001", "003", "005"]);

console.log('✅ All tests completed!');
console.log('\n💡 Verification points:');
console.log('- avatar_set is always preserved');
console.log('- IDs in exemptList are preserved');
console.log('- Other components are removed');
console.log('- ID format is numeric strings ("001", "002", etc.)');