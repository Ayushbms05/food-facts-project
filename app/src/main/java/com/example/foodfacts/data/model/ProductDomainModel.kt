package com.example.foodfacts.data.model

data class ProductDomainModel(
    val barcode: String,
    val name: String,
    val brand: String,
    val imageUrl: String?,
    val nutriScore: String?, // A, B, C, D, E
    val novaGroup: Int?,     // 1, 2, 3, 4
    val ecoScore: String?,   // A, B, C, D, E
    val ingredients: String?,
    val categories: String?,
    val allergens: String?,
    val additives: String? = null,
    val energyKcal: Double?,
    val fat: Double?,
    val saturatedFat: Double?,
    val carbohydrates: Double?,
    val sugars: Double?,
    val fiber: Double?,
    val proteins: Double?,
    val salt: Double?,
    val dataSource: String = "Open Food Facts Database",
    val lastAccessedAt: Long = System.currentTimeMillis()
)
