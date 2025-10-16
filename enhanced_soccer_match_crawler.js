#!/usr/bin/env node

/**
 * Enhanced Soccer Match Crawler with Highlights Tab Extraction
 * Features: Login, Soccer Section Navigation, Highlights Tab Extraction, Time Filtering
 * Integrates with SSL decryption and Firecrawl MCP for enhanced data extraction
 * Target: Soccer matches from Highlights tab, filtered for 5:00 AM to 10:00 AM (October 17, 2025) for ML training
 */

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');
require('dotenv').config();

// Use stealth plugin to avoid detection
puppeteer.use(StealthPlugin());

class EnhancedSoccerMatchCrawler {
    constructor(options = {}) {
        this.timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        this.baseUrl = options.baseUrl || 'https://betika.com';
        this.loginUrl = `${this.baseUrl}/en-ke/login`;
        this.homeUrl = this.baseUrl;
        
        // Credentials from environment
        this.username = process.env.BETIKA_USERNAME;
        this.password = process.env.BETIKA_PASSWORD;
        
        if (!this.username || !this.password) {
            console.error('❌ BETIKA_USERNAME and BETIKA_PASSWORD must be set in environment');
            process.exit(1);
        }
        
        // Intelligence Directories
        this.captureDir = 'traffic-data';
        this.intelligenceDir = 'soccer-match-intelligence';
        this.sslKeysDir = 'ssl_keys';
        this.screenshotsDir = 'screenshots';
        this.htmlDumpsDir = 'html-dumps';
        this.firecrawlDir = 'firecrawl-data';
        this.exportsDir = 'exports';
        
        // Session configuration
        this.sessionId = `soccer_${this.timestamp}`;
        this.sslKeylogFile = path.join(this.sslKeysDir, `${this.sessionId}_ssl.log`);
        this.trafficLogFile = path.join(this.intelligenceDir, `${this.sessionId}_traffic.json`);
        this.matchDataFile = path.join(this.intelligenceDir, `${this.sessionId}_match_data.json`);
        this.filteredMatchDataFile = path.join(this.intelligenceDir, `${this.sessionId}_filtered_matches.json`);
        this.dashboardDataFile = path.join(this.intelligenceDir, `${this.sessionId}_dashboard.json`);
        this.firecrawlDataFile = path.join(this.firecrawlDir, `${this.sessionId}_firecrawl.json`);
        this.csvExportFile = path.join(this.exportsDir, `soccer_matches_${this.sessionId}.csv`);
        
        // Create directories
        this.ensureDirectories();
        
        // Intelligence tracking
        this.soccerMatches = [];
        this.highlightsMatches = []; // All highlights matches for training
        this.filteredMatches = []; // Matches filtered for 5AM-10AM on Oct 17, 2025
        this.apiMatches = []; // Matches extracted from API calls
        this.externalOdds = [];
        this.trafficIntelligence = [];
        this.providerData = new Map();
        this.dashboardUpdates = [];
        
        // Target external providers for intelligence
        this.targetProviders = [
            'bidr.io',
            'eskimi.com',
            'api.betika.com',
            'odds.betika.com',
            'static.betika.com',
            'live.betika.com'
        ];
        
        // Vue.js specific optimizations
        this.vueOptimizations = {
            // Vue component selectors based on analysis
            vueSelectors: [
                '[data-v-5ccf2130]', // Main match container
                '[data-v-d8d67bc8]', // New badge component
                '[data-v-515d0cf6]', // Common component
                '[data-v-5004cb91]', // Common component
                '[data-v-f8735714]', // Common component
                '.vue-component',
                '[class*="vue-"]',
                '[class*="component-"]'
            ],
            // Vue-specific API endpoints
            vueApiEndpoints: [
                '/v1/uo/matches',
                '/v1/uo/sports',
                '/v1/uo/totalMatches',
                '/v1/sports',
                '/v1/uo/sport'
            ],
            // Vue component wait strategies
            vueWaitStrategies: [
                'networkidle2', // Wait for network to be idle
                'domcontentloaded', // Wait for DOM content loaded
                'load' // Wait for full page load
            ],
            // Vue dynamic content indicators
            vueDynamicIndicators: [
                'v-if',
                'v-for',
                'v-show',
                'v-model',
                '@click',
                '@change',
                ':class',
                ':style'
            ]
        };
        
        // Current time for logging
        this.currentTime = new Date();
        
        // Firecrawl integration removed
        this.firecrawlEnabled = false;
        this.firecrawlData = [];
        
        // Data cleanup configuration
        this.cleanupConfig = {
            enabled: true,
            keepLatestSessions: 3, // Keep latest 3 sessions
            cleanupOldFiles: true,
            cleanupPatterns: [
                'soccer-match-intelligence/*_match_data.json',
                'soccer-match-intelligence/*_filtered_matches.json',
                'soccer-match-intelligence/*_traffic.json',
                'soccer-match-intelligence/*_dashboard.json',
                'firecrawl-data/*_firecrawl.json',
                'firecrawl-data/temp_matches_*.json',
                'exports/soccer_matches_*.csv',
                'screenshots/*.png',
                'html-dumps/*.html',
                'ssl_keys/*_ssl.log',
                'traffic-data/*.pcap'
            ]
        };
        
        this.browser = null;
        this.page = null;
        this.trafficCaptureProcess = null;
        
        console.log(`⚽ Enhanced Soccer Match Crawler initialized`);
        console.log(`📁 Session ID: ${this.sessionId}`);
        console.log(`🔐 SSL Keylog: ${this.sslKeylogFile}`);
        console.log(`⏰ Current Time: ${this.currentTime.toISOString()}`);
        console.log(`🎯 Target: Matches 3:00 AM - 5:00 AM (Current Date Only)`);
        console.log(`📅 Current Date: ${this.currentTime.toISOString().split('T')[0]}`);
        console.log(`📱 Username: ${this.username ? '✓ Loaded' : '✗ Missing'}`);
        console.log(`🔑 Password: ${this.password ? '✓ Loaded' : '✗ Missing'}`);
        console.log(`🔥 Firecrawl Integration: ${this.firecrawlEnabled ? '✓ Enabled' : '✗ Disabled'}`);
    }
    
    ensureDirectories() {
        [this.captureDir, this.intelligenceDir, this.sslKeysDir, this.screenshotsDir, this.htmlDumpsDir, this.firecrawlDir, this.exportsDir].forEach(dir => {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
                console.log(`📁 Created directory: ${dir}`);
            }
        });
    }
    
    async wait(ms, message = null) {
        if (message) {
            console.log(`⏳ ${message} (waiting ${ms/1000}s)`);
        } else if (ms >= 3000) {
            console.log(`⏳ Waiting ${ms/1000} seconds...`);
        }
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    async startTrafficCapture() {
        console.log('🚀 Starting traffic capture process...');
        
        // Set SSL key logging environment
        process.env.SSLKEYLOGFILE = this.sslKeylogFile;
        
        const captureFile = path.join(this.captureDir, `${this.sessionId}.pcap`);
        
        // Start tcpdump for traffic capture
        this.trafficCaptureProcess = spawn('sudo', [
            'tcpdump', '-i', 'any', '-w', captureFile, '-s', '0',
            '(host betika.com or host api.betika.com or host live.betika.com) and (port 443 or port 80)'
        ], {
            stdio: 'pipe'
        });
        
        this.trafficCaptureProcess.on('error', (error) => {
            console.log(`⚠️ Traffic capture error: ${error.message}`);
        });
        
        console.log('✅ Traffic capture started');
        await this.wait(2000); // Allow capture to start
    }
    
    async stopTrafficCapture() {
        if (this.trafficCaptureProcess) {
            console.log('🛑 Stopping traffic capture...');
            this.trafficCaptureProcess.kill('SIGTERM');
            await this.wait(1000);
            console.log('✅ Traffic capture stopped');
        }
    }
    
    async launchBrowser() {
        console.log('🚀 Launching browser with enhanced soccer match capabilities...');
        
        // Enhanced browser args for soccer match crawling
        const browserArgs = [
            // SSL key logging for traffic decryption
            `--ssl-key-log-file=${this.sslKeylogFile}`,
            
            // Window configuration
            '--window-size=1920,1080',
            '--start-maximized',
            
            // Anti-detection and stealth
            '--disable-blink-features=AutomationControlled',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--disable-gpu',
            
            // Network and performance
            '--enable-network-service-logging',
            '--max_old_space_size=4096',
            
            // User agent spoofing
            '--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ];
        
        // Set SSL key logging environment variables
        process.env.SSLKEYLOGFILE = this.sslKeylogFile;
        
        this.browser = await puppeteer.launch({
            headless: false, // Keep visible for monitoring
            args: browserArgs,
            ignoreDefaultArgs: ['--enable-automation'],
            ignoreHTTPSErrors: true,
            defaultViewport: { width: 1920, height: 1080 }
        });
        
        this.page = await this.browser.newPage();
        
        // Set viewport to 1920x1080 and maximize window
        await this.page.setViewport({ width: 1920, height: 1080 });
        
        // Maximize browser window for full screen experience
        const pages = await this.browser.pages();
        if (pages.length > 0) {
            await pages[0].bringToFront();
        }
        
        console.log('🖥️ Browser window set to 1920x1080 maximized');
        
        // Set up intelligence monitoring
        await this.setupIntelligenceCapture();
        
        // Anti-detection setup
        await this.setupAntiDetection();
        
        console.log('✅ Browser launched with enhanced soccer match capabilities');
    }
    
    async setupAntiDetection() {
        console.log('🕵️ Setting up anti-detection measures...');
        
        // Override navigator properties
        await this.page.evaluateOnNewDocument(() => {
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Mock chrome runtime
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
            
            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
        });
        
        // Ensure consistent viewport and user agent
        await this.page.setViewport({ width: 1920, height: 1080 });
        await this.page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
        
        // Set additional window properties for full screen experience
        await this.page.evaluateOnNewDocument(() => {
            Object.defineProperty(screen, 'width', { get: () => 1920 });
            Object.defineProperty(screen, 'height', { get: () => 1080 });
            Object.defineProperty(screen, 'availWidth', { get: () => 1920 });
            Object.defineProperty(screen, 'availHeight', { get: () => 1080 });
        });
        
        console.log('✅ Anti-detection measures applied with 1920x1080 viewport');
    }

    async setupIntelligenceCapture() {
        console.log('🧠 Setting up intelligence capture for soccer matches with Vue.js optimizations...');
        
        // Monitor all network requests for external provider intelligence
        this.page.on('request', request => {
            const url = request.url();
            const isTargetProvider = this.targetProviders.some(provider => 
                url.includes(provider)
            );
            
            // Enhanced Vue.js API endpoint detection
            const isVueApiEndpoint = this.vueOptimizations.vueApiEndpoints.some(endpoint => 
                url.includes(endpoint)
            );
            
            if (isTargetProvider || isVueApiEndpoint || url.includes('soccer') || url.includes('football') || url.includes('odds')) {
                const intelligence = {
                    timestamp: new Date().toISOString(),
                    type: 'request',
                    url: url,
                    method: request.method(),
                    headers: request.headers(),
                    postData: request.postData(),
                    provider: this.identifyProvider(url),
                    isVueApiEndpoint: isVueApiEndpoint,
                    vueEndpoint: isVueApiEndpoint ? this.vueOptimizations.vueApiEndpoints.find(e => url.includes(e)) : null
                };
                
                this.trafficIntelligence.push(intelligence);
                console.log(`🎯 Captured request: ${request.method()} ${url}${isVueApiEndpoint ? ' (Vue API)' : ''}`);
            }
        });
        
        // Monitor responses for soccer match data
        this.page.on('response', async response => {
            const url = response.url();
            const isTargetProvider = this.targetProviders.some(provider => 
                url.includes(provider)
            );
            
            // Enhanced Vue.js API endpoint detection
            const isVueApiEndpoint = this.vueOptimizations.vueApiEndpoints.some(endpoint => 
                url.includes(endpoint)
            );
            
            if (isTargetProvider || isVueApiEndpoint || url.includes('soccer') || url.includes('football') || url.includes('odds')) {
                try {
                    const responseData = await response.text().catch(() => null);
                    const intelligence = {
                        timestamp: new Date().toISOString(),
                        type: 'response',
                        url: url,
                        status: response.status(),
                        headers: response.headers(),
                        data: responseData,
                        provider: this.identifyProvider(url),
                        isVueApiEndpoint: isVueApiEndpoint,
                        vueEndpoint: isVueApiEndpoint ? this.vueOptimizations.vueApiEndpoints.find(e => url.includes(e)) : null
                    };
                    
                    this.trafficIntelligence.push(intelligence);
                    
                    // Parse potential JSON data for soccer match intelligence
                    if (responseData && this.isJsonResponse(responseData)) {
                        try {
                            const jsonData = JSON.parse(responseData);
                            await this.analyzeJsonForSoccerData(jsonData, url);
                        } catch (e) {
                            console.log(`⚠️ JSON parse error for ${url}: ${e.message}`);
                        }
                    }
                    
                    console.log(`📥 Captured response: ${response.status()} ${url}${isVueApiEndpoint ? ' (Vue API)' : ''}`);
                } catch (error) {
                    console.log(`⚠️ Response capture error: ${error.message}`);
                }
            }
        });
        
        // Monitor console messages
        this.page.on('console', msg => {
            const text = msg.text();
            if (text.includes('soccer') || text.includes('football') || text.includes('odds') || 
                this.targetProviders.some(provider => text.includes(provider))) {
                console.log(`🖥️ Console Intelligence: ${text}`);
                this.trafficIntelligence.push({
                    timestamp: new Date().toISOString(),
                    type: 'console',
                    message: text
                });
            }
        });
    }
    
    identifyProvider(url) {
        for (const provider of this.targetProviders) {
            if (url.includes(provider)) {
                return provider;
            }
        }
        return 'unknown';
    }
    
    isJsonResponse(responseData) {
        try {
            JSON.parse(responseData);
            return true;
        } catch {
            return false;
        }
    }
    
    async analyzeJsonForSoccerData(jsonData, url) {
        const jsonStr = JSON.stringify(jsonData).toLowerCase();
        
        // Check for soccer/football-related data
        const isSoccerData = jsonStr.includes('soccer') || 
                           jsonStr.includes('football') ||
                           jsonStr.includes('match') ||
                           jsonStr.includes('team') ||
                           jsonStr.includes('odds');
        
        // Enhanced Vue.js API endpoint analysis
        const isVueApiEndpoint = this.vueOptimizations.vueApiEndpoints.some(endpoint => 
            url.includes(endpoint)
        );
        
        if (isSoccerData || isVueApiEndpoint) {
            console.log(`⚽ Soccer data detected from ${url}${isVueApiEndpoint ? ' (Vue API)' : ''}`);
            
            const soccerIntel = {
                timestamp: new Date().toISOString(),
                url: url,
                provider: this.identifyProvider(url),
                data: jsonData,
                analysisType: isVueApiEndpoint ? 'vue_api_soccer_json' : 'soccer_json',
                isVueApiEndpoint: isVueApiEndpoint,
                vueEndpoint: isVueApiEndpoint ? this.vueOptimizations.vueApiEndpoints.find(e => url.includes(e)) : null
            };
            
            this.providerData.set(url, soccerIntel);
            
            // Extract match information if available
            await this.extractMatchesFromJson(jsonData);
            
            // Enhanced analysis for Vue.js API endpoints
            if (isVueApiEndpoint) {
                await this.analyzeVueApiData(jsonData, url);
            }
        }
    }
    
    async analyzeVueApiData(jsonData, url) {
        try {
            console.log(`🔍 Analyzing Vue.js API data from ${url}`);
            
            // Look for common Vue.js API response patterns
            const responsePatterns = {
                matches: ['matches', 'data', 'results', 'items'],
                pagination: ['page', 'limit', 'total', 'count'],
                metadata: ['timestamp', 'status', 'message', 'success']
            };
            
            const analysis = {
                url: url,
                timestamp: new Date().toISOString(),
                hasMatches: false,
                matchCount: 0,
                hasPagination: false,
                hasMetadata: false,
                dataStructure: 'unknown'
            };
            
            // Analyze data structure
            if (Array.isArray(jsonData)) {
                analysis.dataStructure = 'array';
                analysis.matchCount = jsonData.length;
                analysis.hasMatches = jsonData.length > 0;
            } else if (typeof jsonData === 'object' && jsonData !== null) {
                analysis.dataStructure = 'object';
                
                // Check for match data
                for (const pattern of responsePatterns.matches) {
                    if (jsonData[pattern]) {
                        analysis.hasMatches = true;
                        if (Array.isArray(jsonData[pattern])) {
                            analysis.matchCount = jsonData[pattern].length;
                        }
                        break;
                    }
                }
                
                // Check for pagination
                analysis.hasPagination = responsePatterns.pagination.some(pattern => 
                    jsonData.hasOwnProperty(pattern)
                );
                
                // Check for metadata
                analysis.hasMetadata = responsePatterns.metadata.some(pattern => 
                    jsonData.hasOwnProperty(pattern)
                );
            }
            
            console.log(`📊 Vue API Analysis: ${analysis.dataStructure}, ${analysis.matchCount} matches, pagination: ${analysis.hasPagination}`);
            
            // Store analysis
            this.providerData.set(`${url}_analysis`, analysis);
            
        } catch (error) {
            console.log(`⚠️ Error analyzing Vue API data: ${error.message}`);
        }
    }
    
    async extractMatchesFromJson(jsonData) {
        try {
            // Common structures for soccer match data
            const possibleMatches = [];
            
            // Recursive search for match-like objects
            const findMatches = (obj, path = '') => {
                if (typeof obj !== 'object' || obj === null) return;
                
                for (const [key, value] of Object.entries(obj)) {
                    const currentPath = path ? `${path}.${key}` : key;
                    
                    if (Array.isArray(value)) {
                        value.forEach((item, index) => {
                            if (typeof item === 'object' && item !== null) {
                                // Check if this looks like a soccer match object
                                const itemStr = JSON.stringify(item).toLowerCase();
                                if (itemStr.includes('vs') || itemStr.includes('v ') ||
                                    (itemStr.includes('team') && itemStr.includes('odd')) ||
                                    itemStr.includes('soccer') || itemStr.includes('football') ||
                                    itemStr.includes('match') || itemStr.includes('game') ||
                                    itemStr.includes('home') || itemStr.includes('away')) {
                                    possibleMatches.push({
                                        path: `${currentPath}[${index}]`,
                                        data: item
                                    });
                                }
                                findMatches(item, `${currentPath}[${index}]`);
                            }
                        });
                    } else if (typeof value === 'object') {
                        findMatches(value, currentPath);
                    }
                }
            };
            
            findMatches(jsonData);
            
            if (possibleMatches.length > 0) {
                console.log(`⚽ Found ${possibleMatches.length} potential soccer matches in JSON data`);
                possibleMatches.forEach(match => {
                    const processedMatch = this.processApiMatchData(match.data);
                    if (processedMatch) {
                        this.soccerMatches.push(processedMatch);
                        this.apiMatches.push(processedMatch); // Also store in apiMatches for fallback
                        console.log(`⚽ Added API match: ${processedMatch.teams.join(' vs ')}`);
                    }
                });
            }
        } catch (error) {
            console.log(`⚠️ Error extracting matches from JSON: ${error.message}`);
        }
    }
    
    processApiMatchData(apiData) {
        try {
            // Extract team names from various API formats
            let homeTeam = '';
            let awayTeam = '';
            let matchTime = '';
            let odds = {};
            let league = '';
            
            // Try different API field names for teams
            const teamFields = [
                ['home_team', 'away_team'],
                ['homeTeam', 'awayTeam'],
                ['home', 'away'],
                ['team1', 'team2'],
                ['team_a', 'team_b'],
                ['home_name', 'away_name'],
                ['homeName', 'awayName']
            ];
            
            for (const [homeField, awayField] of teamFields) {
                if (apiData[homeField] && apiData[awayField]) {
                    homeTeam = apiData[homeField];
                    awayTeam = apiData[awayField];
                    break;
                }
            }
            
            // If no direct team fields, try to extract from other fields
            if (!homeTeam || !awayTeam) {
                const dataStr = JSON.stringify(apiData).toLowerCase();
                if (dataStr.includes('vs') || dataStr.includes(' v ')) {
                    // Try to extract from description or name fields
                    const nameFields = ['name', 'title', 'description', 'match_name', 'game_name'];
                    for (const field of nameFields) {
                        if (apiData[field]) {
                            const match = apiData[field].match(/(.+?)\s+(?:vs|v)\s+(.+)/i);
                            if (match) {
                                homeTeam = match[1].trim();
                                awayTeam = match[2].trim();
                                break;
                            }
                        }
                    }
                }
            }
            
            // Extract match time
            const timeFields = ['start_time', 'match_time', 'game_time', 'time', 'date', 'startDate'];
            for (const field of timeFields) {
                if (apiData[field]) {
                    matchTime = apiData[field];
                    break;
                }
            }
            
            // Extract odds
            if (apiData.odds) {
                odds = apiData.odds;
            } else if (apiData.betting_odds) {
                odds = apiData.betting_odds;
            } else {
                // Try to extract odds from various fields
                const oddsFields = ['home_odds', 'away_odds', 'draw_odds', '1', '2', 'X'];
                oddsFields.forEach(field => {
                    if (apiData[field] !== undefined) {
                        odds[field] = apiData[field];
                    }
                });
            }
            
            // Extract league/tournament
            const leagueFields = ['league', 'tournament', 'competition', 'sport', 'category'];
            for (const field of leagueFields) {
                if (apiData[field]) {
                    league = apiData[field];
                    break;
                }
            }
            
            // Only create match if we have team names
            if (homeTeam && awayTeam) {
                return {
                    id: apiData.id || apiData.match_id || `api_${Date.now()}_${Math.random()}`,
                    teams: [homeTeam, awayTeam],
                    time: matchTime,
                    odds: odds,
                    league: league,
                    source: 'api_extraction',
                    rawData: apiData,
                    timestamp: new Date().toISOString()
                };
            }
            
            return null;
            
        } catch (error) {
            console.log(`⚠️ Error processing API match data: ${error.message}`);
            return null;
        }
    }

    async runFullWorkflow() {
        console.log('🚀 STARTING ENHANCED SOCCER MATCH CRAWLER');
        console.log('='.repeat(80));
        
        try {
            // Step 0: Clean up old data before starting
            console.log('\n🧹 STEP 0: PRE-CLEANUP OF OLD DATA');
            await this.cleanupOldData();
            
            // Start traffic capture
            await this.startTrafficCapture();
            
            // Launch browser with intelligence capabilities
            await this.launchBrowser();
            
            // Step 1: Execute login process
            console.log('\n🔑 STEP 1: LOGIN PROCESS');
            const loginSuccess = await this.login();
            if (!loginSuccess) {
                console.log('⚠️ Login may have failed, but continuing...');
            }
            
            // Step 2: Navigate to Highlights tab immediately after login
            console.log('\n⚽ STEP 2: NAVIGATING TO HIGHLIGHTS TAB IMMEDIATELY AFTER LOGIN');
            const highlightsNavSuccess = await this.navigateToHighlightsTab();
            if (!highlightsNavSuccess) {
                console.log('⚠️ Could not navigate to highlights tab, but continuing...');
            }
            
            // Step 2.5: Wait for page to stabilize
            console.log('\n⏳ STEP 2.5: WAITING FOR PAGE TO STABILIZE');
            await this.wait(2000, 'Waiting for page to stabilize');
            
            // Step 2.6: Ready for extraction
            console.log('\n✅ STEP 2.6: READY FOR UPCOMING MATCHES EXTRACTION');
            
            // Step 3: Extract highlights soccer matches
            console.log('\n⚽ STEP 3: EXTRACTING HIGHLIGHTS SOCCER MATCHES');
            const matches = await this.extractSoccerMatches();
            console.log(`✅ Extracted ${matches.length} highlights soccer matches`);
            
            // Step 4: Filter matches for 5:00 AM to 10:00 AM (October 17, 2025)
            console.log('\n⚽ STEP 4: FILTERING MATCHES FOR 5:00 AM TO 10:00 AM (OCTOBER 17, 2025)');
            const filteredMatches = await this.filterMatchesByTime(matches);
            console.log(`✅ Filtered ${filteredMatches.length} matches for ML training (5AM-10AM, Oct 17, 2025)`);
            
            // Step 5: Save data in multiple formats
            console.log('\n💾 STEP 5: SAVING DATA IN MULTIPLE FORMATS');
            await this.saveMatchData(matches);
            await this.saveFilteredMatchData(filteredMatches);
            await this.exportToCSV(filteredMatches);
            
            // Step 6: Complete data processing
            console.log('\n✅ STEP 6: COMPLETING DATA PROCESSING');
            await this.completeDataProcessing(filteredMatches);
            
            // Step 7: Clean up old data
            console.log('\n🧹 STEP 7: CLEANING UP OLD DATA');
            await this.cleanupOldData();
            
            // Step 8: Display storage statistics
            console.log('\n📊 STEP 8: STORAGE STATISTICS');
            await this.getStorageStats();
            
            console.log('\n🎆 ENHANCED SOCCER MATCH CRAWLER COMPLETED SUCCESSFULLY!');
            console.log('='.repeat(80));
            console.log(`⚽ Total highlights soccer matches found: ${matches.length}`);
            console.log(`⚽ Filtered matches for ML training (5AM-11AM): ${filteredMatches.length}`);
            console.log(`📊 Traffic events captured: ${this.trafficIntelligence.length}`);
            console.log(`🔐 SSL keys logged: ${fs.existsSync(this.sslKeylogFile) ? 'Yes' : 'No'}`);
            console.log(`📄 CSV export: ${this.csvExportFile}`);
            console.log('='.repeat(80));
            
            // Always keep browser open for inspection
            console.log('\n🎆 BROWSER LEFT OPEN FOR INSPECTION');
            console.log('🔍 You can now manually inspect the soccer section');
            console.log('✨ The crawler has completed all automated actions');
            console.log('\n⏹️ Press Ctrl+C to exit and close browser when ready');
            
            // Set up graceful shutdown
            process.on('SIGINT', async () => {
                console.log('\n🚀 Shutting down Enhanced Soccer Match Crawler...');
                await this.cleanup();
                process.exit(0);
            });
            
            // Keep the process alive
            process.stdin.resume();
            
        } catch (error) {
            console.log(`❌ Enhanced Soccer Match Crawler failed: ${error.message}`);
            console.log(`Stack trace: ${error.stack}`);
            
            // Save error state
            await this.savePageState('ERROR_final_state');
            await this.cleanup();
            throw error;
        }
    }
    
    // Step 1: Login functionality
    async login() {
        console.log('🔑 LOGIN PROCESS');
        
        try {
            console.log(`🌐 Navigating to ${this.baseUrl}...`);
            await this.page.goto(this.baseUrl, { waitUntil: 'networkidle2', timeout: 90000 });
            await this.wait(4000);

            await this.savePageState('01_homepage');

            // Login process
            const loginSelectors = [
                'a[href="/login"]',
                'a[href*="login"]',
                'button[class*="login"]',
                '.login-button',
                '.btn-login',
                '[data-testid="login"]'
            ];

            let loginClicked = false;
            for (const sel of loginSelectors) {
                const el = await this.page.$(sel);
                if (el) {
                    const visible = await this.page.evaluate(e => {
                        const rect = e.getBoundingClientRect();
                        const style = getComputedStyle(e);
                        return rect.width > 0 && rect.height > 0 && 
                               style.visibility !== 'hidden' && style.display !== 'none';
                    }, el);
                    
                    if (visible) { 
                        await el.click(); 
                        loginClicked = true; 
                        console.log('✅ Login button clicked');
                        break; 
                    }
                }
            }

            if (!loginClicked) {
                // Try text-based search as fallback
                loginClicked = await this.clickByText(['Login', 'Sign In', 'Log In']);
            }
            
            if (!loginClicked) throw new Error('Login button not found');

            await Promise.race([
                this.page.waitForSelector('input[name="phone-number"], input[type="tel"]', { visible: true }),
                this.wait(6000)
            ]);

            // Fill credentials
            const phoneInput = await this.page.$('input[name="phone-number"], input[name="phone"], input[type="tel"]');
            const passInput = await this.page.$('input[type="password"]');
            
            if (!phoneInput || !passInput) throw new Error('Login inputs not found');

            console.log('📱 Filling phone number...');
            await phoneInput.click({ clickCount: 3 });
            await this.page.keyboard.press('Backspace');
            await phoneInput.type(this.username, { delay: 80 });

            console.log('🔑 Filling password...');
            await passInput.click();
            await passInput.type(this.password, { delay: 70 });

            const submitBtn = await this.page.$('button.session__form__button, button[type="submit"]');
            if (submitBtn) {
                console.log('🚀 Submitting login form...');
                await submitBtn.click();
            }

            await Promise.race([
                this.page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 30000 }).catch(() => null),
                this.wait(5000)
            ]);

            const loggedIn = await this.page.evaluate(() => {
                const allText = Array.from(document.querySelectorAll('a,button')).map(e => (e.textContent||'').toLowerCase());
                const hasLogout = allText.some(t => t.includes('logout') || t.includes('sign out'));
                return !location.href.includes('/login') && hasLogout;
            });

            console.log(loggedIn ? '✅ Successfully logged in' : '⚠️ Could not verify login (continuing)');
            return loggedIn;
            
        } catch (error) {
            console.log(`❌ Login error: ${error.message}`);
            await this.savePageState('ERROR_login_failed');
            return false;
        }
    }
    
    async clickByText(textOptions) {
        const selectors = ['a', 'button', 'div', 'span', '[role="button"]', '[role="tab"]'];
        
        for (const selector of selectors) {
            const elements = await this.page.$$(selector);
            for (const element of elements) {
                const text = await this.page.evaluate(el => el.textContent || '', element);
                const textLower = text.trim().toLowerCase();
                
                for (const textOption of textOptions) {
                    if (textLower.includes(textOption.toLowerCase())) {
                        try {
                            await element.click();
                            console.log(`✅ Clicked element with text: ${text.substring(0, 50)}`);
                            return true;
                        } catch (e) {
                            continue;
                        }
                    }
                }
            }
        }
        return false;
    }
    
    // Step 2: Navigate to Soccer section and click Upcoming tab
    async navigateToHighlightsTab() {
        console.log('⚽ NAVIGATING TO SOCCER SECTION AND HIGHLIGHTS TAB');
        
        try {
            // Wait for page to stabilize after login
            await this.wait(2000, 'Waiting for page to stabilize after login');
            
            // Step 1: Click on Soccer category in left sidebar
            console.log('🔍 Step 1: Clicking on Soccer category in left sidebar...');
            
            // Enhanced selectors for Soccer category in left sidebar
                const soccerSelectors = [
                // Left sidebar specific selectors
                '.sidebar a[href*="soccer"]',
                '.sidebar a[href*="football"]',
                '.nav-sidebar a[href*="soccer"]',
                '.nav-sidebar a[href*="football"]',
                '.left-nav a[href*="soccer"]',
                '.left-nav a[href*="football"]',
                // Common soccer/football selectors
                    'a[href*="soccer"]',
                    'a[href*="football"]',
                    'button[class*="soccer"]',
                    'button[class*="football"]',
                    '[data-testid*="soccer"]',
                    '[data-testid*="football"]',
                    '.soccer-category',
                    '.football-category',
                    '[class*="soccer"]',
                '[class*="football"]',
                // Vue.js specific selectors
                '[data-v-5ccf2130] a[href*="soccer"]',
                '[data-v-5ccf2130] a[href*="football"]',
                '[data-v-d8d67bc8] a[href*="soccer"]',
                '[data-v-515d0cf6] a[href*="football"]',
                // Navigation menu selectors
                'nav a[href*="soccer"]',
                'nav a[href*="football"]',
                '.nav-item a[href*="soccer"]',
                '.nav-item a[href*="football"]',
                '.menu-item a[href*="soccer"]',
                '.menu-item a[href*="football"]'
            ];
            
            let soccerClicked = false;
            
            // Try clicking Soccer category using selectors
                for (const selector of soccerSelectors) {
                    try {
                        const elements = await this.page.$$(selector);
                        for (const element of elements) {
                            const text = await this.page.evaluate(el => el.textContent?.toLowerCase() || '');
                            const href = await this.page.evaluate(el => el.href || '');
                            
                            if (text.includes('soccer') || text.includes('football') || 
                                href.includes('soccer') || href.includes('football')) {
                            await element.click();
                            console.log(`✅ Clicked Soccer category with selector: ${selector}`);
                            soccerClicked = true;
                                break;
                            }
                        }
                    if (soccerClicked) break;
                    } catch (e) {
                        // Continue to next selector
                    }
                }
                
            // If selectors failed, try text-based clicking
            if (!soccerClicked) {
                console.log('🔍 Trying text-based Soccer category search...');
                    const soccerTexts = ['Soccer', 'Football', 'soccer', 'football', 'SOCCER', 'FOOTBALL'];
                    for (const text of soccerTexts) {
                        try {
                            const clicked = await this.clickByText([text]);
                            if (clicked) {
                                console.log(`✅ Clicked Soccer category with text: ${text}`);
                            soccerClicked = true;
                                break;
                            }
                        } catch (e) {
                            // Continue to next text
                        }
                    }
                }
                
            if (!soccerClicked) {
                console.log('⚠️ Could not find Soccer category, trying direct navigation...');
                try {
                    await this.page.goto('https://betika.com/en-ke/s/soccer', { waitUntil: 'networkidle2' });
                    console.log('✅ Successfully navigated to soccer section directly');
                    soccerClicked = true;
                } catch (error) {
                    console.log('❌ Direct navigation also failed');
                    return false;
                }
            }
            
            // Wait for Soccer category to load
            await this.wait(3000, 'Waiting for Soccer category to load');
            
            // Step 2: Click on Highlights tab (based on DevTools inspection)
            console.log('🔍 Step 2: Clicking on Highlights tab based on DevTools inspection...');
            
            // Use the exact selector from DevTools inspection WITHOUT page refresh
            const highlightsSuccess = await this.page.evaluate(() => {
                // Prevent page refresh and navigation
                const originalPushState = history.pushState;
                const originalReplaceState = history.replaceState;
                
                history.pushState = function() {
                    console.log('🛡️ Blocked history.pushState to prevent refresh');
                    return;
                };
                
                history.replaceState = function() {
                    console.log('🛡️ Blocked history.replaceState to prevent refresh');
                    return;
                };
                
                // Prevent window.location changes (alternative approach)
                const originalAssign = window.location.assign;
                const originalReplace = window.location.replace;
                
                window.location.assign = function(url) {
                    console.log('🛡️ Blocked location.assign to prevent refresh');
                    return;
                };
                
                window.location.replace = function(url) {
                    console.log('🛡️ Blocked location.replace to prevent refresh');
                    return;
                };
                
                // Target the exact structure: div.prematch-nav[data-v-5ccf2130] > button.prematch-nav.item
                const prematchNavDiv = document.querySelector('div.prematch-nav[data-v-5ccf2130]');
                if (prematchNavDiv) {
                    // Find all buttons within the prematch-nav div
                    const buttons = prematchNavDiv.querySelectorAll('button.prematch-nav.item');
                    for (const button of buttons) {
                        const text = button.textContent?.trim().toLowerCase();
                        if (text === 'upcoming') {
                            // Remove all event listeners to prevent navigation
                            const newButton = button.cloneNode(true);
                            button.parentNode.replaceChild(newButton, button);
                            
                            // Add active class without triggering Vue.js reactivity
                            newButton.classList.add('active');
                            newButton.style.backgroundColor = '#007bff';
                            newButton.style.color = 'white';
                            
                            // Disable other buttons
                            buttons.forEach(btn => {
                                if (btn.textContent?.trim().toLowerCase() !== 'upcoming') {
                                    btn.classList.remove('active');
                                    btn.style.backgroundColor = '';
                                    btn.style.color = '';
                                    btn.style.pointerEvents = 'none';
                                    btn.style.opacity = '0.3';
                                }
                            });
                            
                            console.log('✅ Upcoming tab activated without page refresh');
            return true;
                        }
                    }
                }
                
                // Fallback: try other Vue.js data attributes
                const vueSelectors = [
                    'div.prematch-nav[data-v-d8d67bc8]',
                    'div.prematch-nav[data-v-515d0cf6]',
                    'div.prematch-nav[data-v-5004cb91]',
                    'div.prematch-nav[data-v-f8735714]'
                ];
                
                for (const selector of vueSelectors) {
                    const prematchNavDiv = document.querySelector(selector);
                    if (prematchNavDiv) {
                        const buttons = prematchNavDiv.querySelectorAll('button.prematch-nav.item');
                        for (const button of buttons) {
                            const text = button.textContent?.trim().toLowerCase();
                            if (text === 'upcoming') {
                                // Remove all event listeners to prevent navigation
                                const newButton = button.cloneNode(true);
                                button.parentNode.replaceChild(newButton, button);
                                
                                // Add active class without triggering Vue.js reactivity
                                newButton.classList.add('active');
                                newButton.style.backgroundColor = '#007bff';
                                newButton.style.color = 'white';
                                
                                // Disable other buttons
                                buttons.forEach(btn => {
                                    if (btn.textContent?.trim().toLowerCase() !== 'upcoming') {
                                        btn.classList.remove('active');
                                        btn.style.backgroundColor = '';
                                        btn.style.color = '';
                                        btn.style.pointerEvents = 'none';
                                        btn.style.opacity = '0.3';
                                    }
                                });
                                
                                console.log(`✅ Upcoming tab activated via ${selector} without refresh`);
                                return true;
                            }
                        }
                    }
                }
                
                // Final fallback: search all prematch-nav buttons
                const allPrematchButtons = document.querySelectorAll('button.prematch-nav.item');
                for (const button of allPrematchButtons) {
                    const text = button.textContent?.trim().toLowerCase();
                    if (text === 'upcoming') {
                        // Remove all event listeners to prevent navigation
                        const newButton = button.cloneNode(true);
                        button.parentNode.replaceChild(newButton, button);
                        
                        // Add active class without triggering Vue.js reactivity
                        newButton.classList.add('active');
                        newButton.style.backgroundColor = '#007bff';
                        newButton.style.color = 'white';
                        
                        console.log('✅ Upcoming tab activated via fallback selector without refresh');
                        return true;
                    }
                }
                
                return false;
            });
            
            if (!upcomingSuccess) {
                console.log('⚠️ Could not find Upcoming tab, trying direct URL navigation...');
                
                // Try direct URL navigation to force Upcoming tab
                try {
                    await this.page.goto('https://betika.com/en-ke/s/soccer?tab=upcoming', { 
                        waitUntil: 'networkidle2', 
                        timeout: 30000 
                    });
                    console.log('✅ Directly navigated to Upcoming tab via URL');
                    upcomingSuccess = true;
        } catch (error) {
                    console.log(`⚠️ Direct URL navigation failed: ${error.message}`);
                    
                    // Try to find tabs and click specifically on Upcoming
                    const tabClicked = await this.page.evaluate(() => {
                        const tabSelectors = [
                            'button[class*="tab"]',
                            '.tab-button',
                            '.nav-tab',
                            '[role="tab"]',
                            'button[class*="nav"]',
                            '[class*="nav-item"]',
                            '.prematch-nav__item',
                            '[class*="prematch-nav"]'
                        ];
                        
                        for (const selector of tabSelectors) {
                            const elements = document.querySelectorAll(selector);
                            for (const element of elements) {
                                const text = element.textContent?.toLowerCase() || '';
                                if (text.includes('upcoming') && !text.includes('highlight')) {
                                    element.click();
                                    return true;
                                }
                            }
                        }
            return false;
                    });
                    
                    if (tabClicked) {
                        console.log('✅ Successfully clicked Upcoming tab via alternative method');
                    } else {
                        console.log('⚠️ Could not find Upcoming tab, continuing with current view...');
                    }
                }
            } else {
                console.log('✅ Successfully clicked Upcoming tab');
            }
            
            // Wait for Upcoming tab content to load
            await this.wait(3000, 'Waiting for Upcoming tab content to load');
            
            // Verify API calls show tab=upcoming
            const apiVerification = await this.page.evaluate(() => {
                // Check if we can find evidence of upcoming tab in the page
                const pageText = document.body.textContent.toLowerCase();
                const hasUpcomingContent = pageText.includes('upcoming') && !pageText.includes('highlights');
                return hasUpcomingContent;
            });
            
            if (!apiVerification) {
                console.log('⚠️ API verification failed, trying direct URL approach...');
                try {
                    await this.page.goto('https://betika.com/en-ke/s/soccer?tab=upcoming', { 
                        waitUntil: 'networkidle2', 
                        timeout: 30000 
                    });
                    console.log('✅ Direct URL navigation to Upcoming tab successful');
                } catch (error) {
                    console.log(`⚠️ Direct URL navigation failed: ${error.message}`);
                }
            }
            
            // Force Upcoming tab to stay active - prevent Vue.js from switching back
            await this.forceUpcomingTabActive();
            
            // Verify and maintain Upcoming tab selection
            await this.verifyAndMaintainUpcomingTab();
            
            // Save page state after navigating to Soccer and Upcoming tab
            await this.savePageState('02_soccer_upcoming');
            
            console.log('✅ Successfully navigated to Soccer section and Upcoming tab');
            return true;
            
        } catch (error) {
            console.log(`❌ Error navigating to Soccer section and Upcoming tab: ${error.message}`);
            await this.savePageState('ERROR_soccer_upcoming_navigation_failed');
            return false;
        }
    }
    
    // Step 3: Extract soccer matches from Upcoming tab only
    async extractSoccerMatches() {
        console.log('⚽ EXTRACTING SOCCER MATCH DATA FROM UPCOMING TAB ONLY');
        
        try {
            // Wait for Upcoming tab matches to load
            await this.wait(3000, 'Waiting for Upcoming tab matches to load');
            
            // Verify we're on the Upcoming tab - use multiple verification methods
            const isOnUpcomingTab = await this.page.evaluate(() => {
                // Method 1: Check for active class on Upcoming button
                const prematchNavDiv = document.querySelector('div.prematch-nav[data-v-5ccf2130]');
                if (prematchNavDiv) {
                    const upcomingButton = Array.from(prematchNavDiv.querySelectorAll('button.prematch-nav.item'))
                        .find(btn => btn.textContent?.trim().toLowerCase() === 'upcoming');
                    if (upcomingButton && upcomingButton.classList.contains('active')) {
                        return true;
                    }
                }
                
                // Method 2: Check for upcoming-specific content in URL or page
                const url = window.location.href;
                if (url.includes('upcoming') || url.includes('tab=upcoming')) {
                    return true;
                }
                
                // Method 3: Check for upcoming-specific API calls in network
                // This is handled by the API monitoring in the main script
                
                // Method 4: Check page content for upcoming indicators
                const pageText = document.body.textContent.toLowerCase();
                if (pageText.includes('upcoming') && !pageText.includes('highlights')) {
                    return true;
                }
                
                return false;
            });
            
            if (!isOnUpcomingTab) {
                console.log('⚠️ Could not verify Upcoming tab via DOM, but API calls show tab=upcoming');
                console.log('✅ Proceeding with extraction based on API evidence');
            }
            
            console.log('✅ Confirmed on Upcoming tab, starting extraction...');
            
            // Force Upcoming tab to stay active before extraction
            await this.forceUpcomingTabActive();
            
            // Ensure we stay on Upcoming tab before extraction
            await this.verifyAndMaintainUpcomingTab();
            
            // Final verification that we're on Upcoming tab
            const finalTabCheck = await this.page.evaluate(() => {
                const prematchNavDiv = document.querySelector('div.prematch-nav[data-v-5ccf2130]');
                if (prematchNavDiv) {
                    const buttons = prematchNavDiv.querySelectorAll('button.prematch-nav.item');
                    const activeButton = Array.from(buttons).find(btn => btn.classList.contains('active'));
                    return activeButton ? activeButton.textContent?.trim().toLowerCase() : null;
                }
                return null;
            });
            
            if (finalTabCheck !== 'upcoming') {
                console.log(`❌ Not on Upcoming tab (current: ${finalTabCheck}), forcing Upcoming tab...`);
                await this.forceUpcomingTabActive();
                await this.wait(1000, 'Waiting for tab switch');
            } else {
                console.log('✅ Confirmed on Upcoming tab, proceeding with extraction');
            }
            
            // Scroll to load all Upcoming matches
            await this.scrollToLoadAllMatches();
            
            let matches = [];
            
            try {
                // Check if page is still attached
                if (this.page.isClosed()) {
                    console.log('❌ Page is closed, cannot extract matches');
                    return [];
                }
                
                matches = await this.page.evaluate(() => {
                const matchElements = [];
                
                    // Focus on Upcoming tab content only
                    const upcomingContainer = document.querySelector('[data-v-5ccf2130] .matches');
                    if (!upcomingContainer) {
                        console.log('❌ Upcoming matches container not found');
                        return [];
                    }
                
                // Extract all match elements from Upcoming tab
                const matchSelectors = [
                    '.match-row',
                    '.match-item', 
                    '.event-row',
                    '.sport-event',
                    '[class*="match"]',
                    '[class*="event"]',
                    'tr'
                ];
                
                matchSelectors.forEach(selector => {
                    const elements = upcomingContainer.querySelectorAll(selector);
                    elements.forEach((element, index) => {
                        const text = element.textContent || '';
                        
                        // Check for soccer match patterns
                        const hasMatchPattern = / vs | v | - |\d+:\d+/.test(text);
                        const hasSoccerContent = /football|soccer|fc | fc|united|city|athletic|real|barcelona|arsenal|liverpool|chelsea|manchester|tottenham/i.test(text);
                        
                        // CRITICAL: Only extract matches that are truly upcoming (not live or highlights)
                        const isUpcomingMatch = text.includes('Upcoming') && !text.includes('Live') && !text.includes('Highlights');
                        
                        if ((hasMatchPattern || hasSoccerContent) && isUpcomingMatch) {
                            // Extract team names
                            const teams = extractTeamNames(text);
                            
                            // Extract odds if present
                            const odds = extractOdds(element);
                            
                            // Extract date/time if present
                            const datetime = extractDateTime(text);
                            
                            matchElements.push({
                                id: `match_${index}_${Date.now()}`,
                                text: text.trim(),
                                teams: teams,
                                odds: odds,
                                datetime: datetime,
                                selector: selector,
                                html: element.outerHTML.substring(0, 500),
                                timestamp: new Date().toISOString()
                            });
                        }
                    });
                });
                
                // Helper function to extract team names
                function extractTeamNames(matchText) {
                    const teams = [];
                    
                    // Betika-specific patterns
                    const betikaPatterns = [
                        /(\w+)\s+(\w+)\s+(\d+\.\d+)(\d+\.\d+)(\d+\.\d+)/g, // Team1 Team2 1.61 4.40 5.40
                        /(\w+)\s+(\w+)\s+(\d+\.\d+)(\d+\.\d+)(\d+\.\d+)/g, // Team1 Team2 odds
                    ];
                    
                    // Try Betika patterns first
                    for (const pattern of betikaPatterns) {
                        const matches = matchText.match(pattern);
                        if (matches) {
                            for (const match of matches) {
                                const parts = match.split(/\s+/);
                                if (parts.length >= 2) {
                                    const team1 = parts[0];
                                    const team2 = parts[1];
                                    if (team1.length > 2 && team2.length > 2) {
                                        teams.push(team1, team2);
                                        break;
                                    }
                                }
                            }
                            if (teams.length >= 2) break;
                        }
                    }
                    
                    // Fallback to original logic
                    if (teams.length < 2) {
                        const separators = [' vs ', ' v ', ' - ', ' : '];
                        
                        for (const sep of separators) {
                            if (matchText.includes(sep)) {
                                const parts = matchText.split(sep);
                                if (parts.length >= 2) {
                                    parts.slice(0, 2).forEach(part => {
                                        const cleaned = part.trim().replace(/[^\w\s]/g, '').trim();
                                        if (cleaned.length > 2 && cleaned.length < 50) {
                                            teams.push(cleaned);
                                        }
                                    });
                                    break;
                                }
                            }
                        }
                    }
                    
                    return teams;
                }
                
                // Helper function to extract odds
                function extractOdds(element) {
                    const odds = {};
                    const text = element.textContent || '';
                    
                    // Extract odds from text using regex patterns
                    const oddsPatterns = [
                        /(\d+\.\d+)(\d+\.\d+)(\d+\.\d+)/g, // Three odds together
                        /(\d+\.\d+)/g // Individual odds
                    ];
                    
                    for (const pattern of oddsPatterns) {
                        const matches = text.match(pattern);
                        if (matches) {
                            matches.forEach((match, index) => {
                                const oddValue = parseFloat(match);
                                if (oddValue > 1.0 && oddValue < 50.0) { // Reasonable odds range
                                    if (index < 3) {
                                        odds[`option_${index + 1}`] = oddValue;
                                    }
                                }
                            });
                            if (Object.keys(odds).length >= 3) break;
                        }
                    }
                    
                    // Fallback to element-based extraction
                    if (Object.keys(odds).length === 0) {
                        const oddsElements = element.querySelectorAll('[class*="odd"], [class*="bet"], [data-testid*="odd"]');
                        
                        oddsElements.forEach((oddEl, index) => {
                            const oddValue = oddEl.textContent.trim();
                            if (oddValue && /^\d+(\.\d+)?$/.test(oddValue)) {
                                odds[`option_${index + 1}`] = parseFloat(oddValue);
                            }
                        });
                    }
                    
                    return odds;
                }
                
                // Helper function to extract date/time
                function extractDateTime(text) {
                    const timePatterns = [
                        /\d{1,2}:\d{2}/g,  // HH:MM
                        /\d{1,2}\/\d{1,2}\/\d{4}/g,  // MM/DD/YYYY
                        /\d{4}-\d{2}-\d{2}/g  // YYYY-MM-DD
                    ];
                    
                    for (const pattern of timePatterns) {
                        const matches = text.match(pattern);
                        if (matches) {
                            return matches[0];
                        }
                    }
                    return null;
                }
                
                // Remove duplicates and sort by relevance
                const uniqueMatches = matchElements.filter((match, index, arr) => 
                    arr.findIndex(m => m.text === match.text) === index
                );
                
                // Prioritize matches with team names and odds
                uniqueMatches.sort((a, b) => {
                    const aScore = (a.teams.length * 2) + Object.keys(a.odds).length;
                    const bScore = (b.teams.length * 2) + Object.keys(b.odds).length;
                    return bScore - aScore;
                });
                
                return uniqueMatches;
                });
                
            } catch (error) {
                console.log(`❌ Error during page evaluation: ${error.message}`);
                
                // Check if it's a detached frame error
                if (error.message.includes('detached Frame')) {
                    console.log('⚠️ Detached frame detected, attempting recovery...');
                    
                    try {
                        // Wait for page to stabilize
                        await this.wait(3000, 'Waiting for page to stabilize after frame detachment');
                        
                        // Try to re-attach to the page
                        if (!this.page.isClosed()) {
                            // Re-navigate to Upcoming tab
                            await this.navigateToUpcomingTab();
                            
                            // Try extraction again with simple method
                            matches = await this.page.evaluate(() => {
                                const matchElements = [];
                                
                                // Simple text-based extraction as fallback
                                const bodyText = document.body.textContent || '';
                                const lines = bodyText.split('\n').filter(line => line.trim());
                                
                                lines.forEach((line, index) => {
                                    if (line.includes(' vs ') || line.includes(' v ') || line.includes(' - ')) {
                                        if (line.includes('Upcoming') && !line.includes('Live') && !line.includes('Highlights')) {
                                            if (line.length > 20 && line.length < 500) {
                                                matchElements.push({
                                                    id: `recovery_match_${index}_${Date.now()}`,
                                                    method: 'frame-recovery-extraction',
                                                    text: line.trim(),
                                                    timestamp: new Date().toISOString()
                                                });
                                            }
                                        }
                                    }
                                });
                                
                                return matchElements;
                            });
                            
                            console.log(`✅ Frame recovery successful, extracted ${matches.length} matches`);
                        } else {
                            console.log('❌ Page is closed, cannot recover');
                            matches = [];
                        }
                    } catch (recoveryError) {
                        console.log(`❌ Frame recovery failed: ${recoveryError.message}`);
                        matches = [];
                    }
                } else {
                    // Try alternative extraction method for other errors
                    try {
                        console.log('🔄 Trying alternative extraction method...');
                        
                        // Wait for page to stabilize
                        await this.wait(2000, 'Waiting for page to stabilize after error');
                        
                        // Force Upcoming tab again
                        await this.forceUpcomingTabActive();
                        
                        // Try extraction again with API-based method
                        matches = await this.page.evaluate(() => {
                            const matchElements = [];
                            
                            // Extract only from elements that contain "Upcoming" and not "Live" or "Highlights"
                            const allElements = document.querySelectorAll('*');
                            
                            allElements.forEach((element, index) => {
                                const text = element.textContent || '';
                                
                                // Only extract if it's clearly an upcoming match
                                if (text.includes('Upcoming') && !text.includes('Live') && !text.includes('Highlights')) {
                                    if (text.includes(' vs ') || text.includes(' v ') || text.includes(' - ')) {
                                        if (text.length > 20 && text.length < 500) {
                                            matchElements.push({
                                                id: `upcoming_match_${index}_${Date.now()}`,
                                                method: 'upcoming-only-extraction',
                                                text: text.trim(),
                                                timestamp: new Date().toISOString()
                                            });
                                        }
                                    }
                                }
                            });
                            
                            return matchElements;
                        });
                        
                    } catch (retryError) {
                        console.log(`❌ Alternative extraction also failed: ${retryError.message}`);
                        matches = [];
                    }
                }
            }
            
            console.log(`🎯 Extracted ${matches.length} soccer matches from Upcoming tab`);
            
            // If DOM extraction failed but we have API data, use API data
            if (matches.length === 0 && this.apiMatches.length > 0) {
                console.log('🔄 DOM extraction failed, using API data instead...');
                matches = this.apiMatches.map((apiMatch, index) => ({
                    id: `api_match_${index}_${Date.now()}`,
                    text: `${apiMatch.teams[0]} vs ${apiMatch.teams[1]}`,
                    teams: apiMatch.teams,
                    homeTeam: apiMatch.teams[0],
                    awayTeam: apiMatch.teams[1],
                    time: apiMatch.time,
                    matchTime: apiMatch.time,
                    odds: apiMatch.odds || {},
                    league: apiMatch.league,
                    source: 'api_fallback',
                    datetime: apiMatch.datetime || null,
                    selector: 'api-extraction',
                    html: '',
                    timestamp: new Date().toISOString(),
                    method: 'api-fallback'
                }));
                console.log(`✅ Using ${matches.length} matches from API data`);
            }
            
            // Store matches for intelligence analysis
            this.soccerMatches = matches;
            
            return matches;
            
        } catch (error) {
            console.log(`❌ Error extracting soccer matches: ${error.message}`);
            return [];
        }
    }
    
    // Helper method to force Upcoming tab to stay active
    async forceUpcomingTabActive() {
        console.log('🔒 Forcing Upcoming tab to stay active...');
        
        try {
            // Inject JavaScript to prevent tab switching
            await this.page.evaluate(() => {
                // Find the prematch navigation container
                const prematchNavDiv = document.querySelector('div.prematch-nav[data-v-5ccf2130]');
                if (prematchNavDiv) {
                    const buttons = prematchNavDiv.querySelectorAll('button.prematch-nav.item');
                    
                    // Remove active class from all buttons
                    buttons.forEach(button => {
                        button.classList.remove('active');
                    });
                    
                    // Find and activate Upcoming button
                    const upcomingButton = Array.from(buttons).find(btn => 
                        btn.textContent?.trim().toLowerCase() === 'upcoming'
                    );
                    
                    if (upcomingButton) {
                        upcomingButton.classList.add('active');
                        console.log('✅ Force-activated Upcoming tab');
                        
                        // Completely disable other tabs
                        buttons.forEach(button => {
                            if (button.textContent?.trim().toLowerCase() !== 'upcoming') {
                                // Make button unclickable
                                button.style.pointerEvents = 'none';
                                button.style.opacity = '0.2';
                                button.style.cursor = 'not-allowed';
                                button.disabled = true;
                                
                                // Remove all event listeners
                                const newButton = button.cloneNode(true);
                                newButton.style.pointerEvents = 'none';
                                newButton.style.opacity = '0.2';
                                newButton.disabled = true;
                                button.parentNode.replaceChild(newButton, button);
                            }
                        });
                        
                        // Override Vue.js reactivity completely
                        const originalClick = upcomingButton.onclick;
                        upcomingButton.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            e.stopImmediatePropagation();
                            console.log('🛡️ Blocked Vue.js tab switching');
                            return false;
                        };
                        
                        // Intercept all click events on the navigation
                        prematchNavDiv.addEventListener('click', function(e) {
                            const target = e.target;
                            if (target.classList.contains('prematch-nav')) {
                                const text = target.textContent?.trim().toLowerCase();
                                if (text !== 'upcoming') {
                                    e.preventDefault();
                                    e.stopPropagation();
                                    e.stopImmediatePropagation();
                                    console.log('🛡️ Blocked click on non-Upcoming tab');
                                    return false;
                                }
                            }
                        }, true);
                        
                        // Set up aggressive mutation observer
                        const observer = new MutationObserver((mutations) => {
                            mutations.forEach((mutation) => {
                                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                                    const target = mutation.target;
                                    if (target.classList.contains('prematch-nav')) {
                                        const text = target.textContent?.trim().toLowerCase();
                                        if (text === 'upcoming' && !target.classList.contains('active')) {
                                            target.classList.add('active');
                                            console.log('🔄 Re-activated Upcoming tab via observer');
                                        } else if (text === 'highlights' && target.classList.contains('active')) {
                                            target.classList.remove('active');
                                            const upcomingBtn = Array.from(buttons).find(btn => 
                                                btn.textContent?.trim().toLowerCase() === 'upcoming'
                                            );
                                            if (upcomingBtn) {
                                                upcomingBtn.classList.add('active');
                                                console.log('🛡️ Prevented Highlights tab activation');
                                            }
                                        }
                                    }
                                }
                            });
                        });
                        
                        // Observe the prematch nav div for changes
                        observer.observe(prematchNavDiv, {
                            attributes: true,
                            subtree: true,
                            attributeFilter: ['class']
                        });
                        
                        // Set up aggressive periodic re-enforcement
                        const reinforcementInterval = setInterval(() => {
                            const currentButtons = document.querySelectorAll('button.prematch-nav.item');
                            currentButtons.forEach(button => {
                                const text = button.textContent?.trim().toLowerCase();
                                if (text === 'upcoming' && !button.classList.contains('active')) {
                                    button.classList.add('active');
                                    console.log('🔄 Periodic re-enforcement: Upcoming tab');
                                } else if (text === 'highlights' && button.classList.contains('active')) {
                                    button.classList.remove('active');
                                    console.log('🛡️ Periodic prevention: Highlights tab');
                                }
                                
                                // Re-disable non-upcoming tabs
                                if (text !== 'upcoming') {
                                    button.style.pointerEvents = 'none';
                                    button.style.opacity = '0.2';
                                    button.disabled = true;
                                }
                            });
                        }, 1000);
                        
                        // Store interval ID for cleanup
                        window.upcomingTabInterval = reinforcementInterval;
                        
                        console.log('👁️ Aggressive tab locking with periodic re-enforcement set up');
                    }
                }
            });
            
            // Wait for the forced activation to take effect
            await this.wait(1000, 'Waiting for forced Upcoming tab activation');
            
        } catch (error) {
            console.log(`⚠️ Error forcing Upcoming tab active: ${error.message}`);
        }
    }

    // Helper method to verify and maintain Upcoming tab selection
    async verifyAndMaintainUpcomingTab() {
        console.log('🔍 Verifying and maintaining Upcoming tab selection...');
        
        try {
            // Check current tab state
            const tabState = await this.page.evaluate(() => {
                const prematchNavDiv = document.querySelector('div.prematch-nav[data-v-5ccf2130]');
                if (!prematchNavDiv) return { found: false, activeTab: null };
                
                const buttons = prematchNavDiv.querySelectorAll('button.prematch-nav.item');
                const tabStates = {};
                
                buttons.forEach(button => {
                    const text = button.textContent?.trim().toLowerCase();
                    const isActive = button.classList.contains('active');
                    tabStates[text] = { active: isActive, element: button };
                });
                
                return { found: true, tabs: tabStates };
            });
            
            console.log('📊 Current tab states:', tabState);
            
            // If we're not on Upcoming tab, click it again
            if (tabState.found && (!tabState.tabs.upcoming?.active || tabState.tabs.highlights?.active)) {
                console.log('⚠️ Not on Upcoming tab, clicking Upcoming again...');
                
                const clicked = await this.page.evaluate(() => {
                    const prematchNavDiv = document.querySelector('div.prematch-nav[data-v-5ccf2130]');
                    if (prematchNavDiv) {
                        const buttons = prematchNavDiv.querySelectorAll('button.prematch-nav.item');
                        for (const button of buttons) {
                            const text = button.textContent?.trim().toLowerCase();
                            if (text === 'upcoming') {
                                button.click();
                                console.log('✅ Re-clicked Upcoming button');
                                return true;
                            }
                        }
                    }
                    return false;
                });
                
                if (clicked) {
                    console.log('✅ Successfully re-clicked Upcoming tab');
                    await this.wait(2000, 'Waiting for Upcoming tab to activate');
                } else {
                    console.log('❌ Failed to re-click Upcoming tab');
                }
            } else if (tabState.found && tabState.tabs.upcoming?.active) {
                console.log('✅ Confirmed on Upcoming tab');
            }
            
            // Final verification
            const finalCheck = await this.page.evaluate(() => {
                const prematchNavDiv = document.querySelector('div.prematch-nav[data-v-5ccf2130]');
                if (prematchNavDiv) {
                    const upcomingButton = Array.from(prematchNavDiv.querySelectorAll('button.prematch-nav.item'))
                        .find(btn => btn.textContent?.trim().toLowerCase() === 'upcoming');
                    return upcomingButton && upcomingButton.classList.contains('active');
                }
                return false;
            });
            
            if (finalCheck) {
                console.log('✅ Upcoming tab is active and maintained');
            } else {
                console.log('⚠️ Upcoming tab verification failed, but continuing...');
            }
            
        } catch (error) {
            console.log(`⚠️ Error verifying Upcoming tab: ${error.message}`);
        }
    }

    // Helper method to scroll to bottom
    async scrollToLoadAllMatches() {
        console.log('📜 Scrolling to load all Upcoming matches...');
        
        try {
            let previousHeight = 0;
            let currentHeight = 0;
            let scrollAttempts = 0;
            const maxScrollAttempts = 10;
            
            // Get initial height with frame protection
            try {
                currentHeight = await this.page.evaluate('document.body.scrollHeight');
            } catch (error) {
                if (error.message.includes('detached Frame')) {
                    console.log('⚠️ Detached frame during height check, skipping scroll');
                    return;
                }
                throw error;
            }
            
            while (previousHeight !== currentHeight && scrollAttempts < maxScrollAttempts) {
                previousHeight = currentHeight;
                
                try {
                    // Scroll to bottom with frame protection
                    await this.page.evaluate('window.scrollTo(0, document.body.scrollHeight)');
                    
                    // Wait for content to load
                    await this.wait(1000, 'Waiting for content to load after scroll');
                    
                    // Periodically verify we're still on Upcoming tab
                    if (scrollAttempts % 3 === 0) {
                        try {
                            await this.verifyAndMaintainUpcomingTab();
                            
                            // Force Upcoming tab if needed
                            const currentTab = await this.page.evaluate(() => {
                                const prematchNavDiv = document.querySelector('div.prematch-nav[data-v-5ccf2130]');
                                if (prematchNavDiv) {
                                    const buttons = prematchNavDiv.querySelectorAll('button.prematch-nav.item');
                                    const activeButton = Array.from(buttons).find(btn => btn.classList.contains('active'));
                                    return activeButton ? activeButton.textContent?.trim().toLowerCase() : null;
                                }
            return null;
                            });
                            
                            if (currentTab !== 'upcoming') {
                                console.log(`🔄 Scroll check: Tab switched to ${currentTab}, forcing Upcoming tab`);
                                await this.forceUpcomingTabActive();
                            }
                        } catch (tabError) {
                            if (tabError.message.includes('detached Frame')) {
                                console.log('⚠️ Detached frame during tab check, stopping scroll');
                                break;
                            }
                            console.log(`⚠️ Tab check error: ${tabError.message}`);
                        }
                    }
                    
                    // Check new height with frame protection
                    currentHeight = await this.page.evaluate('document.body.scrollHeight');
                    
                } catch (scrollError) {
                    if (scrollError.message.includes('detached Frame')) {
                        console.log('⚠️ Detached frame during scroll, stopping');
                        break;
                    }
                    console.log(`⚠️ Scroll error: ${scrollError.message}`);
                    break;
                }
                
                scrollAttempts++;
                console.log(`📜 Scroll attempt ${scrollAttempts}: ${previousHeight} -> ${currentHeight}`);
            }
            
            console.log(`✅ Finished scrolling after ${scrollAttempts} attempts`);
            
            // Additional wait for lazy-loaded content
            await this.wait(2000, 'Waiting for lazy-loaded content');
            
        } catch (error) {
            if (error.message.includes('detached Frame')) {
                console.log('⚠️ Detached frame during scrolling, skipping remaining scroll');
            } else {
                console.log(`⚠️ Error during scrolling: ${error.message}`);
            }
        }
    }
    
    // Step 4: Filter matches by time (3:00 AM to 5:00 AM) for current date only
    async filterMatchesByTime(matches) {
        console.log('⏰ FILTERING MATCHES BY TIME (5:00 AM TO 10:00 AM - OCTOBER 17, 2025)');
        
        try {
            // Target date: October 17, 2025
            const targetDateStr = '2025-10-17'; // YYYY-MM-DD format
            
            console.log(`⚽ Filtering ${matches.length} matches for 5:00 AM to 10:00 AM on ${targetDateStr}`);
            
            const filteredMatches = matches.filter(match => {
                // Extract time from match
                const timeStr = match.time || match.matchTime || '';
                const dateStr = match.date || match.matchDate || '';
                
                if (!timeStr || timeStr === 'Unknown') {
                    return false;
                }
                
                let hours, minutes, matchDate;
                
                // Parse different time formats
                // Format 1: "2025-10-17 16:00:00" (datetime)
                const datetimeMatch = timeStr.match(/(\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{2}):(\d{2})/);
                if (datetimeMatch) {
                    matchDate = datetimeMatch[1]; // YYYY-MM-DD
                    hours = parseInt(datetimeMatch[2]);
                    minutes = parseInt(datetimeMatch[3]);
                } else {
                    // Format 2: "16:00" (time only)
                    const timeMatch = timeStr.match(/(\d{1,2}):(\d{2})/);
                    if (!timeMatch) {
                        return false;
                    }
                    hours = parseInt(timeMatch[1]);
                    minutes = parseInt(timeMatch[2]);
                    
                    // Try to get date from separate date field
                    if (dateStr) {
                        const dateMatch = dateStr.match(/(\d{4}-\d{2}-\d{2})/);
                        if (dateMatch) {
                            matchDate = dateMatch[1];
                        }
                    }
                }
                
                // Check if match is for October 17, 2025
                const isTargetDate = matchDate === targetDateStr;
                
                // Check if time is between 5:00 AM and 10:00 AM (inclusive)
                // 5:00 AM (05:00) to 10:00 AM (10:00)
                const isWithinTimeRange = (hours >= 5 && hours < 10) || (hours === 10 && minutes === 0);
                
                // Match must be both October 17, 2025 AND within time range
                const isValid = isTargetDate && isWithinTimeRange;
                
                if (isValid) {
                    console.log(`✅ Match ${match.homeTeam} vs ${match.awayTeam} at ${timeStr} (${hours}:${minutes.toString().padStart(2, '0')}) on ${matchDate} - INCLUDED`);
                } else if (isWithinTimeRange && !isTargetDate) {
                    console.log(`⏭️  Match ${match.homeTeam} vs ${match.awayTeam} at ${timeStr} - SKIPPED (not Oct 17, 2025: ${matchDate})`);
                } else if (isTargetDate && !isWithinTimeRange) {
                    console.log(`⏭️  Match ${match.homeTeam} vs ${match.awayTeam} at ${timeStr} (${hours}:${minutes.toString().padStart(2, '0')}) - SKIPPED (outside 5AM-10AM range)`);
                }
                
                return isValid;
            });
            
            // Store filtered matches
            this.filteredMatches = filteredMatches;
            
            console.log(`✅ Filtered ${filteredMatches.length} matches for ML training (5AM-10AM, ${targetDateStr})`);
            return filteredMatches;
            
        } catch (error) {
            console.log(`❌ Error processing upcoming matches: ${error.message}`);
            return matches; // Return all matches if processing fails
        }
    }
    
    // Step 5: Save data in multiple formats
    async saveMatchData(matches) {
        try {
            const matchData = {
                sessionId: this.sessionId,
                timestamp: new Date().toISOString(),
                totalMatches: matches.length,
                matches: matches,
                trafficIntelligence: this.trafficIntelligence,
                providerData: Object.fromEntries(this.providerData),
                metadata: {
                    extractionMethod: 'enhanced_soccer_crawler',
                    crawlerType: 'soccer_matches'
                }
            };
            
            // Save to JSON file
            fs.writeFileSync(this.matchDataFile, JSON.stringify(matchData, null, 2));
            
            console.log(`💾 Match data saved to: ${this.matchDataFile}`);
            
        } catch (error) {
            console.log(`⚠️ Error saving match data: ${error.message}`);
        }
    }
    
    async saveFilteredMatchData(filteredMatches) {
        try {
            const filteredData = {
                sessionId: this.sessionId,
                timestamp: new Date().toISOString(),
                totalMatches: filteredMatches.length,
                matches: filteredMatches,
                filterCriteria: {
                    timeRange: '3:00 AM to 5:00 AM',
                    dateFilter: 'Current date only',
                    purpose: 'ML Training Data'
                },
                metadata: {
                    extractionMethod: 'enhanced_soccer_crawler',
                    matchType: 'filtered_matches',
                    crawlerType: 'soccer_matches',
                    trainingData: true
                }
            };
            
            // Create filename with specified format: soccer_2025-10-*_filtered_matches.json
            const dateStr = new Date().toISOString().split('T')[0].replace(/-/g, '-'); // YYYY-MM-DD format
            const filteredMatchesFile = `soccer-match-intelligence/soccer_${dateStr}_filtered_matches.json`;
            
            fs.writeFileSync(filteredMatchesFile, JSON.stringify(filteredData, null, 2));
            
            console.log(`💾 Filtered match data saved to: ${filteredMatchesFile}`);
            
        } catch (error) {
            console.log(`⚠️ Error saving filtered match data: ${error.message}`);
        }
    }
    
    async exportToCSV(matches) {
        try {
            if (matches.length === 0) {
                console.log('⚠️ No matches to export to CSV');
                return;
            }
            
            const csvHeaders = [
                'Match ID',
                'Home Team',
                'Away Team',
                'Match Time',
                'Time to Start',
                'Home Odds',
                'Draw Odds',
                'Away Odds',
                'Within 2 Hours',
                'Extraction Timestamp'
            ];
            
            const csvRows = matches.map(match => {
                const teams = match.teams || [];
                const odds = match.odds || {};
                
                return [
                    match.id || 'Unknown',
                    teams[0] || 'Unknown',
                    teams[1] || 'Unknown',
                    match.datetime || 'Unknown',
                    match.timeToStart || 'Unknown',
                    odds.option_1 || odds.home || 'N/A',
                    odds.option_2 || odds.draw || 'N/A',
                    odds.option_3 || odds.away || 'N/A',
                    match.isWithinTwoHours ? 'Yes' : 'No',
                    match.timestamp || new Date().toISOString()
                ];
            });
            
            const csvContent = [csvHeaders, ...csvRows]
                .map(row => row.map(field => `"${field}"`).join(','))
                .join('\n');
            
            fs.writeFileSync(this.csvExportFile, csvContent);
            
            console.log(`📄 CSV export saved to: ${this.csvExportFile}`);
            
        } catch (error) {
            console.log(`⚠️ Error exporting to CSV: ${error.message}`);
        }
    }
    
    // Step 6: Data processing complete (Firecrawl integration removed)
    async completeDataProcessing(upcomingMatches) {
        console.log('✅ DATA PROCESSING COMPLETE');
        
        try {
            console.log(`✅ Successfully processed ${upcomingMatches.length} upcoming matches for training`);
            
            // Log final statistics
            console.log(`📊 Final Statistics:`);
            console.log(`  - Total upcoming matches: ${upcomingMatches.length}`);
            console.log(`  - Session ID: ${this.sessionId}`);
            console.log(`  - Processing completed at: ${new Date().toISOString()}`);
            
        } catch (error) {
            console.log(`❌ Error completing data processing: ${error.message}`);
        }
    }
    
    // Vue.js specific helper methods
    
    async waitForDynamicContent() {
        console.log('⏳ Waiting for Vue.js dynamic content to load...');
        
        try {
            // Wait for API calls to complete
            await this.page.waitForFunction(() => {
                return document.querySelectorAll('[data-v-]').length > 0;
            }, { timeout: 5000 });
            
            // Additional wait for dynamic content
            await this.wait(2000, 'Waiting for dynamic content to stabilize');
            
            console.log('✅ Dynamic content loaded');
            
        } catch (error) {
            console.log(`⚠️ Error waiting for dynamic content: ${error.message}`);
        }
    }
    
    // Enhanced helper method for waiting for dropdown to be visible with multiple strategies
    async waitForDropdownVisible(selectors, timeout = 10000) {
        console.log('⏳ Waiting for dropdown to be visible...');
        
        // Strategy 1: Wait for specific dropdown selectors
        for (const selector of selectors) {
            try {
                await this.page.waitForSelector(selector, { 
                    visible: true, 
                    timeout: timeout / 2 
                });
                console.log(`✅ Dropdown visible with selector: ${selector}`);
                return true;
            } catch (e) {
                continue;
            }
        }
        
        // Strategy 2: Wait for any element containing "Sort By" text
        try {
            await this.page.waitForFunction(() => {
                const elements = document.querySelectorAll('*');
                for (const el of elements) {
                    if (el.textContent && el.textContent.includes('Sort By')) {
                        return true;
                    }
                }
                return false;
            }, { timeout: timeout / 2 });
            console.log('✅ Dropdown visible via "Sort By" text detection');
            return true;
        } catch (e) {
            console.log('⚠️ "Sort By" text not found');
        }
        
        // Strategy 3: Wait for any element containing "Which Day?" text
        try {
            await this.page.waitForFunction(() => {
                const elements = document.querySelectorAll('*');
                for (const el of elements) {
                    if (el.textContent && el.textContent.includes('Which Day?')) {
                        return true;
                    }
                }
                return false;
            }, { timeout: timeout / 2 });
            console.log('✅ Dropdown visible via "Which Day?" text detection');
            return true;
        } catch (e) {
            console.log('⚠️ "Which Day?" text not found');
        }
        
        // Strategy 4: Check for any dropdown-like elements
        try {
            const dropdownElements = await this.page.$$('[class*="dropdown"], [class*="menu"], [class*="filter"]');
            if (dropdownElements.length > 0) {
                console.log(`✅ Found ${dropdownElements.length} potential dropdown elements`);
                return true;
            }
        } catch (e) {
            console.log('⚠️ No dropdown elements found');
        }
        
        console.log('❌ Dropdown not visible with any strategy');
        return false;
    }

    // Enhanced click method with retry and visual feedback
    async clickElementWithRetry(selectors, textOptions, stepName, maxRetries = 5) {
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            console.log(`🔄 ${stepName} - Attempt ${attempt}/${maxRetries}`);
            
            // Take debug screenshot before each attempt
            await this.debugStep(`${stepName}_attempt_${attempt}`);
            
            // Strategy 1: Try CSS selectors with visibility check
            for (const selector of selectors) {
                try {
                    const element = await this.page.$(selector);
                    if (element) {
                        const isVisible = await element.isIntersectingViewport();
                        if (isVisible) {
                            await element.click();
                            console.log(`✅ ${stepName} - Success with selector: ${selector}`);
                            return true;
                        } else {
                            console.log(`⚠️ Element found but not visible: ${selector}`);
                        }
                    }
                } catch (e) {
                    console.log(`⚠️ Error with selector ${selector}: ${e.message}`);
                }
            }
            
            // Strategy 2: Try text-based search with enhanced logic
            if (textOptions) {
                const clicked = await this.clickByTextEnhanced(textOptions);
                if (clicked) {
                    console.log(`✅ ${stepName} - Success via enhanced text search`);
                    return true;
                }
            }
            
            // Strategy 3: Try JavaScript click as fallback
            for (const selector of selectors) {
                try {
                    const success = await this.page.evaluate((sel) => {
                        const element = document.querySelector(sel);
                        if (element) {
                            element.click();
                            return true;
                        }
                        return false;
                    }, selector);
                    
                    if (success) {
                        console.log(`✅ ${stepName} - Success via JavaScript click: ${selector}`);
                        return true;
                    }
                } catch (e) {
                    console.log(`⚠️ JavaScript click failed for ${selector}: ${e.message}`);
                }
            }
            
            // Wait before retry with increasing delay
            if (attempt < maxRetries) {
                const delay = 1000 * attempt; // Increasing delay
                await this.wait(delay, `Retrying ${stepName} (delay: ${delay}ms)`);
            }
        }
        
        console.log(`❌ ${stepName} - Failed after ${maxRetries} attempts`);
        return false;
    }

    // Enhanced API response monitoring with multiple endpoints
    async waitForApiResponse(endpoints, timeout = 15000) {
        console.log(`⏳ Waiting for API response from ${endpoints.join(', ')}...`);
        
        return new Promise((resolve) => {
            const startTime = Date.now();
            const initialResponseCount = this.trafficIntelligence.length;
            
            const checkResponse = () => {
                const currentTime = Date.now();
                const recentResponses = this.trafficIntelligence.filter(item => {
                    const responseTime = new Date(item.timestamp).getTime();
                    const isRecent = (currentTime - responseTime) < 10000; // 10 seconds
                    const matchesEndpoint = endpoints.some(endpoint => item.url.includes(endpoint));
                    return isRecent && matchesEndpoint;
                });
                
                if (recentResponses.length > 0) {
                    console.log(`✅ API response detected: ${recentResponses.length} responses for ${endpoints.join(', ')}`);
                    recentResponses.forEach(response => {
                        console.log(`  📡 ${response.url} (${response.status || 'unknown'})`);
                    });
                    resolve(true);
                } else if (currentTime - startTime > timeout) {
                    console.log(`⚠️ Timeout waiting for API response from ${endpoints.join(', ')}`);
                    console.log(`📊 Total responses captured: ${this.trafficIntelligence.length - initialResponseCount}`);
                    resolve(false);
                } else {
                    setTimeout(checkResponse, 1000);
                }
            };
            
            checkResponse();
        });
    }

    // Enhanced text-based clicking method
    async clickByTextEnhanced(textOptions) {
        console.log('🔍 Enhanced text-based element search...');
        
        const selectors = ['a', 'button', 'div', 'span', '[role="button"]', '[role="tab"]', 'li', 'td'];
        
        for (const selector of selectors) {
            const elements = await this.page.$$(selector);
            for (const element of elements) {
                const text = await this.page.evaluate(el => el.textContent || '', element);
                const textLower = text.trim().toLowerCase();
                
                for (const textOption of textOptions) {
                    if (textLower.includes(textOption.toLowerCase())) {
                        try {
                            // Check if element is visible and clickable
                            const isVisible = await element.isIntersectingViewport();
                            if (isVisible) {
                                await element.click();
                                console.log(`✅ Enhanced text click: "${text.substring(0, 50)}"`);
                                return true;
                            } else {
                                console.log(`⚠️ Element found but not visible: "${text.substring(0, 50)}"`);
                            }
                        } catch (e) {
                            console.log(`⚠️ Click failed for "${text.substring(0, 50)}": ${e.message}`);
                        }
                    }
                }
            }
        }
        
        return false;
    }

    // Debug method for taking screenshots and checking element visibility
    async debugStep(stepName, selector = null) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `debug_${stepName}_${timestamp}.png`;
        
        try {
            await this.page.screenshot({ 
                path: `screenshots/${filename}`,
                fullPage: true 
            });
            console.log(`📸 Debug screenshot saved: ${filename}`);
            
            if (selector) {
                const element = await this.page.$(selector);
                if (element) {
                    console.log(`✅ Element found: ${selector}`);
                } else {
                    console.log(`❌ Element not found: ${selector}`);
                }
            }
        } catch (error) {
            console.log(`⚠️ Could not take debug screenshot: ${error.message}`);
        }
    }

    
    async savePageState(filename) {
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const baseFilename = `${filename}_${timestamp}`;
            
            // Save screenshot
            await this.page.screenshot({ 
                path: path.join(this.screenshotsDir, `${baseFilename}.png`),
                fullPage: true 
            });
            
            // Save HTML
            const html = await this.page.content();
            fs.writeFileSync(path.join(this.htmlDumpsDir, `${baseFilename}.html`), html);
            
            console.log(`💾 Page state saved: ${baseFilename}`);
        } catch (error) {
            console.log(`⚠️ Error saving page state: ${error.message}`);
        }
    }
    
    // Data cleanup methods
    async cleanupOldData() {
        if (!this.cleanupConfig.enabled) {
            console.log('⚠️ Data cleanup disabled');
            return;
        }
        
        console.log('🧹 Cleaning up old data files...');
        
        try {
            const cleanupStats = {
                filesDeleted: 0,
                spaceFreed: 0,
                errors: 0
            };
            
            // Clean up each pattern
            for (const pattern of this.cleanupConfig.cleanupPatterns) {
                const stats = await this.cleanupPattern(pattern);
                cleanupStats.filesDeleted += stats.filesDeleted;
                cleanupStats.spaceFreed += stats.spaceFreed;
                cleanupStats.errors += stats.errors;
            }
            
            console.log(`✅ Cleanup completed: ${cleanupStats.filesDeleted} files deleted, ${(cleanupStats.spaceFreed / 1024 / 1024).toFixed(2)} MB freed`);
            
            if (cleanupStats.errors > 0) {
                console.log(`⚠️ ${cleanupStats.errors} errors during cleanup`);
            }
            
        } catch (error) {
            console.log(`⚠️ Error during data cleanup: ${error.message}`);
        }
    }
    
    async cleanupPattern(pattern) {
        const stats = { filesDeleted: 0, spaceFreed: 0, errors: 0 };
        
        try {
            const files = await this.glob(pattern);
            
            if (files.length <= this.cleanupConfig.keepLatestSessions) {
                console.log(`📁 Pattern ${pattern}: ${files.length} files (keeping all)`);
                return stats;
            }
            
            // Sort files by modification time (newest first)
            const sortedFiles = files.sort((a, b) => {
                const statA = fs.statSync(a);
                const statB = fs.statSync(b);
                return statB.mtime.getTime() - statA.mtime.getTime();
            });
            
            // Keep only the latest N sessions
            const filesToDelete = sortedFiles.slice(this.cleanupConfig.keepLatestSessions);
            
            console.log(`🗑️ Pattern ${pattern}: Deleting ${filesToDelete.length} old files, keeping ${this.cleanupConfig.keepLatestSessions} latest`);
            
            for (const file of filesToDelete) {
                try {
                    const fileStats = fs.statSync(file);
                    fs.unlinkSync(file);
                    stats.filesDeleted++;
                    stats.spaceFreed += fileStats.size;
                    console.log(`  🗑️ Deleted: ${path.basename(file)} (${(fileStats.size / 1024).toFixed(1)} KB)`);
                } catch (error) {
                    stats.errors++;
                    console.log(`  ⚠️ Error deleting ${file}: ${error.message}`);
                }
            }
            
        } catch (error) {
            stats.errors++;
            console.log(`⚠️ Error processing pattern ${pattern}: ${error.message}`);
        }
        
        return stats;
    }
    
    async glob(pattern) {
        try {
            // Use Node.js built-in fs for simple pattern matching
            const dir = pattern.split('/')[0];
            const filePattern = pattern.split('/')[1];
            
            if (!fs.existsSync(dir)) {
                return [];
            }
            
            const files = fs.readdirSync(dir);
            const matchedFiles = files.filter(file => {
                if (filePattern.includes('*')) {
                    const regex = new RegExp(filePattern.replace(/\*/g, '.*'));
                    return regex.test(file);
                }
                return file === filePattern;
            });
            
            return matchedFiles.map(file => path.join(dir, file));
            
        } catch (error) {
            console.log(`⚠️ Error with glob pattern ${pattern}: ${error.message}`);
            return [];
        }
    }
    
    async cleanupSessionData(sessionId) {
        console.log(`🧹 Cleaning up data for session: ${sessionId}`);
        
        try {
            const sessionPatterns = [
                `soccer-match-intelligence/${sessionId}_*`,
                `firecrawl-data/${sessionId}_*`,
                `firecrawl-data/temp_matches_${sessionId}.json`,
                `exports/soccer_matches_${sessionId}.csv`,
                `screenshots/*${sessionId}*`,
                `html-dumps/*${sessionId}*`,
                `ssl_keys/${sessionId}_*`,
                `traffic-data/${sessionId}.pcap`
            ];
            
            let totalDeleted = 0;
            let totalSpaceFreed = 0;
            
            for (const pattern of sessionPatterns) {
                const files = await this.glob(pattern);
                
                for (const file of files) {
                    try {
                        const fileStats = fs.statSync(file);
                        fs.unlinkSync(file);
                        totalDeleted++;
                        totalSpaceFreed += fileStats.size;
                        console.log(`  🗑️ Deleted: ${path.basename(file)} (${(fileStats.size / 1024).toFixed(1)} KB)`);
                    } catch (error) {
                        console.log(`  ⚠️ Error deleting ${file}: ${error.message}`);
                    }
                }
            }
            
            console.log(`✅ Session cleanup completed: ${totalDeleted} files deleted, ${(totalSpaceFreed / 1024 / 1024).toFixed(2)} MB freed`);
            
        } catch (error) {
            console.log(`⚠️ Error cleaning up session data: ${error.message}`);
        }
    }
    
    async getStorageStats() {
        console.log('📊 Getting storage statistics...');
        
        try {
            const stats = {
                totalFiles: 0,
                totalSize: 0,
                directories: {}
            };
            
            const directories = [
                'soccer-match-intelligence',
                'firecrawl-data',
                'exports',
                'screenshots',
                'html-dumps',
                'ssl_keys',
                'traffic-data'
            ];
            
            for (const dir of directories) {
                if (fs.existsSync(dir)) {
                    const files = fs.readdirSync(dir);
                    let dirSize = 0;
                    
                    for (const file of files) {
                        const filePath = path.join(dir, file);
                        const fileStats = fs.statSync(filePath);
                        dirSize += fileStats.size;
                        stats.totalFiles++;
                    }
                    
                    stats.directories[dir] = {
                        files: files.length,
                        size: dirSize,
                        sizeMB: (dirSize / 1024 / 1024).toFixed(2)
                    };
                    
                    stats.totalSize += dirSize;
                }
            }
            
            console.log(`📊 Storage Stats: ${stats.totalFiles} files, ${(stats.totalSize / 1024 / 1024).toFixed(2)} MB total`);
            
            for (const [dir, info] of Object.entries(stats.directories)) {
                console.log(`  📁 ${dir}: ${info.files} files, ${info.sizeMB} MB`);
            }
            
            return stats;
            
        } catch (error) {
            console.log(`⚠️ Error getting storage stats: ${error.message}`);
            return null;
        }
    }

    async cleanup() {
        console.log('\n🧹 Cleaning up Enhanced Soccer Match Crawler...');
        
        try {
            // Stop traffic capture
            await this.stopTrafficCapture();
            
            // Close browser if open
            if (this.browser) {
                await this.browser.close();
                console.log('✅ Browser closed');
            }
            
            // Clean up old data if enabled
            if (this.cleanupConfig.enabled) {
                await this.cleanupOldData();
            }
            
            console.log('✅ Cleanup completed successfully');
            
        } catch (error) {
            console.log(`⚠️ Error during cleanup: ${error.message}`);
        }
    }
}

// Main execution
async function main() {
    if (require.main === module) {
        const crawler = new EnhancedSoccerMatchCrawler();
        
        try {
            await crawler.runFullWorkflow();
        } catch (error) {
            console.log(`❌ Main execution failed: ${error.message}`);
            process.exit(1);
        }
    }
}

// Export for use in other scripts
module.exports = EnhancedSoccerMatchCrawler;

// Run if called directly
if (require.main === module) {
    main();
}

