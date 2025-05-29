const axios = require('axios');

async function testEmbeddingFix() {
    console.log('Testing embedding grouping fix...');
    
    // Simple test with Nova Energy entries
    const testDescriptions = [
        'nova energy',
        'nova energy onlineeftpos',
        'some other merchant',
        'completely different store'
    ];
    
    try {
        console.log('Sending request to clean_text endpoint...');
        console.log('Request data:', JSON.stringify({
            descriptions: testDescriptions,
            use_embedding_grouping: true,
            embedding_similarity_threshold: 0.6
        }, null, 2));
        
        const response = await axios.post('http://localhost/clean_text', {
            descriptions: testDescriptions,
            use_embedding_grouping: true,
            embedding_similarity_threshold: 0.6
        }, {
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': 'test-api-key-123',
                'X-User-ID': 'test-user-123'
            },
            timeout: 30000
        });
        
        console.log('\n✅ Request successful!');
        console.log(`Response status: ${response.status}`);
        
        console.log('\nFull response data:');
        console.log(JSON.stringify(response.data, null, 2));
        
        const grouped = response.data.groups || {};
        console.log('\nGrouped results:');
        console.log(JSON.stringify(grouped, null, 2));
        
        for (const [canonical, groupItems] of Object.entries(grouped)) {
            console.log(`\nCanonical: ${canonical}`);
            console.log(`  Items: ${groupItems.join(', ')}`);
        }
        
        // Check if Nova Energy entries are grouped together
        const novaGroups = Object.entries(grouped).filter(([canonical, groupItems]) => 
            canonical.toLowerCase().includes('nova') || 
            groupItems.some(item => item.toLowerCase().includes('nova'))
        );
        
        console.log(`\n📊 Found ${novaGroups.length} Nova Energy group(s)`);
        
        if (novaGroups.length === 1) {
            const [canonical, groupItems] = novaGroups[0];
            const hasOriginal = groupItems.includes('nova energy');
            const hasOnline = groupItems.includes('nova energy onlineeftpos');
            
            if (hasOriginal && hasOnline) {
                console.log('🎉 SUCCESS: Nova Energy entries are properly grouped together!');
            } else {
                console.log('❌ PARTIAL: Nova Energy group found but missing expected variations');
                console.log(`   Has 'nova energy': ${hasOriginal}`);
                console.log(`   Has 'nova energy onlineeftpos': ${hasOnline}`);
            }
        } else if (novaGroups.length > 1) {
            console.log('❌ ISSUE: Nova Energy entries are in multiple groups (should be 1)');
        } else {
            console.log('❌ ISSUE: No Nova Energy groups found');
        }
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        if (error.response) {
            console.error('Status:', error.response.status);
            console.error('Response data:', JSON.stringify(error.response.data, null, 2));
        }
        if (error.code) {
            console.error('Error code:', error.code);
        }
    }
}

testEmbeddingFix().catch(console.error);
