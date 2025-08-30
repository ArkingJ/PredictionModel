// Global variables
let allPredictions = [];
let filteredPredictions = [];

// DOM elements
const loadingElement = document.getElementById('loading');
const predictionsContainer = document.getElementById('predictions-container');
const predictionsGrid = document.getElementById('predictions-grid');
const noPredictionsElement = document.getElementById('no-predictions');
const totalMatchesElement = document.getElementById('total-matches');
const lastUpdatedElement = document.getElementById('last-updated');
const avgConfidenceElement = document.getElementById('avg-confidence');
const matchTypeFilter = document.getElementById('match-type-filter');
const outcomeFilter = document.getElementById('outcome-filter');
const confidenceFilter = document.getElementById('confidence-filter');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadPredictions();
    setupEventListeners();
});

// Set up event listeners
function setupEventListeners() {
    matchTypeFilter.addEventListener('change', filterPredictions);
    outcomeFilter.addEventListener('change', filterPredictions);
    confidenceFilter.addEventListener('change', filterPredictions);
}

// Load predictions from JSON file
async function loadPredictions() {
    try {
        showLoading(true);
        
        const response = await fetch('predictions.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        allPredictions = data;
        
        // Update stats
        updateStats();
        
        // Display predictions
        displayPredictions(allPredictions);
        
        showLoading(false);
        
    } catch (error) {
        console.error('Error loading predictions:', error);
        showError('Failed to load predictions. Please try again later.');
        showLoading(false);
    }
}

// Show/hide loading state
function showLoading(show) {
    if (show) {
        loadingElement.style.display = 'block';
        predictionsContainer.style.display = 'none';
        noPredictionsElement.style.display = 'none';
    } else {
        loadingElement.style.display = 'none';
    }
}

// Show error message
function showError(message) {
    noPredictionsElement.style.display = 'block';
    predictionsContainer.style.display = 'none';
    
    const errorContent = `
        <div class="no-data">
            <i class="fas fa-exclamation-triangle"></i>
            <h3>Error Loading Predictions</h3>
            <p>${message}</p>
        </div>
    `;
    
    noPredictionsElement.innerHTML = errorContent;
}

// Update statistics
function updateStats() {
    const upcomingMatches = allPredictions.filter(p => p.match_type === 'upcoming');
    const recentMatches = allPredictions.filter(p => p.match_type === 'recent');
    
    totalMatchesElement.textContent = allPredictions.length;
    
    // Calculate average confidence
    const totalConfidence = allPredictions.reduce((sum, pred) => sum + pred.confidence, 0);
    const avgConfidence = allPredictions.length > 0 ? (totalConfidence / allPredictions.length * 100).toFixed(1) : 0;
    avgConfidenceElement.textContent = `${avgConfidence}%`;
    
    // Set last updated time
    const now = new Date();
    lastUpdatedElement.textContent = now.toLocaleTimeString();
    
    // Update stats card descriptions
    const totalMatchesDesc = document.querySelector('#total-matches').nextElementSibling;
    if (totalMatchesDesc) {
        totalMatchesDesc.textContent = `Total: ${upcomingMatches.length} upcoming, ${recentMatches.length} recent`;
    }
}

// Display predictions
function displayPredictions(predictions) {
    if (predictions.length === 0) {
        noPredictionsElement.style.display = 'block';
        predictionsContainer.style.display = 'none';
        return;
    }
    
    noPredictionsElement.style.display = 'none';
    predictionsContainer.style.display = 'block';
    
    predictionsGrid.innerHTML = '';
    
    predictions.forEach(prediction => {
        const predictionCard = createPredictionCard(prediction);
        predictionsGrid.appendChild(predictionCard);
    });
}

// Create prediction card
function createPredictionCard(prediction) {
    const card = document.createElement('div');
    card.className = 'prediction-card';
    
    // Add match type indicator
    const matchTypeClass = prediction.match_type === 'recent' ? 'recent-match' : 'upcoming-match';
    card.classList.add(matchTypeClass);
    
    // Format date
    const matchDate = new Date(prediction.date);
    const formattedDate = matchDate.toLocaleDateString('en-GB', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
    
    // Get confidence level
    const confidenceLevel = getConfidenceLevel(prediction.confidence);
    
    // Create outcome badge
    const outcomeBadge = createOutcomeBadge(prediction.predicted_outcome);
    
    // Create match type badge
    const matchTypeBadge = prediction.match_type === 'recent' ? 
        '<div class="match-type-badge recent">Recent Result</div>' : 
        '<div class="match-type-badge upcoming">Upcoming Match</div>';
    
    // Add actual result if available
    let actualResultSection = '';
    if (prediction.match_type === 'recent' && prediction.actual_result) {
        const actualResultBadge = createOutcomeBadge(prediction.actual_result);
        actualResultSection = `
            <div class="actual-result-section">
                <div class="actual-result-label">Actual Result:</div>
                <div class="actual-result-badge ${actualResultBadge.class}">
                    ${actualResultBadge.text}
                </div>
                <div class="actual-score">${prediction.goals_home} - ${prediction.goals_away}</div>
            </div>
        `;
    }
    
    card.innerHTML = `
        ${matchTypeBadge}
        
        <div class="match-teams">
            <div class="team">
                <div class="team-name">${prediction.home_team}</div>
                <div class="team-type">Home</div>
            </div>
            <div class="vs">VS</div>
            <div class="team">
                <div class="team-name">${prediction.away_team}</div>
                <div class="team-type">Away</div>
            </div>
        </div>
        
        <div class="match-date">${formattedDate}</div>
        
        <div class="prediction-outcome">
            <div class="outcome-badge ${outcomeBadge.class}">
                ${outcomeBadge.text}
            </div>
        </div>
        
        ${actualResultSection}
        
        <div class="confidence-section">
            <div class="confidence-label">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill ${confidenceLevel}" style="width: ${prediction.confidence * 100}%"></div>
            </div>
        </div>
        
        <div class="probabilities">
            <div class="probability-item">
                <div class="probability-label">Home Win</div>
                <div class="probability-value">${(prediction.predictions.home_win * 100).toFixed(1)}%</div>
            </div>
            <div class="probability-item">
                <div class="probability-label">Draw</div>
                <div class="probability-value">${(prediction.predictions.draw * 100).toFixed(1)}%</div>
            </div>
            <div class="probability-item">
                <div class="probability-label">Away Win</div>
                <div class="probability-value">${(prediction.predictions.away_win * 100).toFixed(1)}%</div>
            </div>
        </div>
    `;
    
    return card;
}

// Create outcome badge
function createOutcomeBadge(outcome) {
    switch (outcome) {
        case 'H':
            return { class: 'home', text: 'Home Win' };
        case 'D':
            return { class: 'draw', text: 'Draw' };
        case 'A':
            return { class: 'away', text: 'Away Win' };
        default:
            return { class: 'draw', text: 'Unknown' };
    }
}

// Get confidence level for styling
function getConfidenceLevel(confidence) {
    if (confidence >= 0.7) return 'high';
    if (confidence >= 0.5) return 'medium';
    return 'low';
}

// Filter predictions
function filterPredictions() {
    const matchTypeFilterValue = matchTypeFilter.value;
    const outcomeFilterValue = outcomeFilter.value;
    const confidenceFilterValue = confidenceFilter.value;
    
    filteredPredictions = allPredictions.filter(prediction => {
        // Match type filter
        if (matchTypeFilterValue !== 'all' && prediction.match_type !== matchTypeFilterValue) {
            return false;
        }
        
        // Outcome filter
        if (outcomeFilterValue !== 'all' && prediction.predicted_outcome !== outcomeFilterValue) {
            return false;
        }
        
        // Confidence filter
        if (confidenceFilterValue !== 'all') {
            const confidence = prediction.confidence;
            switch (confidenceFilterValue) {
                case 'high':
                    if (confidence < 0.7) return false;
                    break;
                case 'medium':
                    if (confidence < 0.5 || confidence >= 0.7) return false;
                    break;
                case 'low':
                    if (confidence >= 0.5) return false;
                    break;
            }
        }
        
        return true;
    });
    
    displayPredictions(filteredPredictions);
}

// Utility function to format percentage
function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}

// Utility function to format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-GB', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Add some interactive features
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add hover effects for prediction cards
    document.addEventListener('mouseover', function(e) {
        if (e.target.closest('.prediction-card')) {
            e.target.closest('.prediction-card').style.transform = 'translateY(-5px)';
        }
    });
    
    document.addEventListener('mouseout', function(e) {
        if (e.target.closest('.prediction-card')) {
            e.target.closest('.prediction-card').style.transform = 'translateY(0)';
        }
    });
});

// Add keyboard navigation support
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        // Reset filters on Escape key
        outcomeFilter.value = 'all';
        confidenceFilter.value = 'all';
        filterPredictions();
    }
});

// Add auto-refresh functionality (optional)
function setupAutoRefresh() {
    // Refresh predictions every 30 minutes
    setInterval(() => {
        loadPredictions();
    }, 30 * 60 * 1000);
}

// Initialize auto-refresh if needed
// setupAutoRefresh();

// Add error handling for network issues
window.addEventListener('online', function() {
    console.log('Network connection restored');
    // Optionally reload predictions when connection is restored
    // loadPredictions();
});

window.addEventListener('offline', function() {
    console.log('Network connection lost');
    showError('Network connection lost. Please check your internet connection.');
});

