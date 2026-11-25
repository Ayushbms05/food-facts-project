package com.example.foodfacts.data.model

data class MetricComparison(
    val metricName: String,
    val valueA: Double?,
    val valueB: Double?,
    val unit: String,
    val lowerIsBetter: Boolean,
    val winner: ComparisonWinner
)

enum class ComparisonWinner {
    PRODUCT_A,
    PRODUCT_B,
    TIE,
    UNKNOWN
}

data class ComparisonResult(
    val productA: ProductDomainModel,
    val productB: ProductDomainModel,
    val metrics: List<MetricComparison>,
    val recommendedProduct: ComparisonWinner,
    val summary: String
)
