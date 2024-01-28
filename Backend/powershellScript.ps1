# Define the API Endpoint
$apiUrl = "https://localhost:7212/Home/AddStockPricesByBatch"

# Call the API using Invoke-WebRequest
try {
    $response = Invoke-WebRequest -Uri $apiUrl -Method Get
    Write-Output "API Response: $($response.Content)"
}
catch {
    Write-Error "API call failed: $_"
}
