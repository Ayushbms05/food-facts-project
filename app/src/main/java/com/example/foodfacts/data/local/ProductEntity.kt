package com.example.foodfacts.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.example.foodfacts.data.model.ProductDomainModel

@Entity(tableName = "products")
data class ProductEntity(
    @PrimaryKey
    val barcode: String,
    val name: String,
    val brand: String,
    val imageUrl: String?,
    val nutriScore: String?,
    val novaGroup: Int?,
    val ecoScore: String?,
    val ingredients: String?,
    val categories: String?,
    val allergens: String?,
    val additives: String?,
    val energyKcal: Double?,
    val fat: Double?,
    val saturatedFat: Double?,
    val carbohydrates: Double?,
    val sugars: Double?,
    val fiber: Double?,
    val proteins: Double?,
    val salt: Double?,
    val dataSource: String?,
    val lastAccessedAt: Long
)

fun ProductEntity.toDomainModel(): ProductDomainModel {
    return ProductDomainModel(
        barcode = barcode,
        name = name,
        brand = brand,
        imageUrl = imageUrl,
        nutriScore = nutriScore,
        novaGroup = novaGroup,
        ecoScore = ecoScore,
        ingredients = ingredients,
        categories = categories,
        allergens = allergens,
        additives = additives,
        energyKcal = energyKcal,
        fat = fat,
        saturatedFat = saturatedFat,
        carbohydrates = carbohydrates,
        sugars = sugars,
        fiber = fiber,
        proteins = proteins,
        salt = salt,
        dataSource = dataSource ?: "Open Food Facts Database",
        lastAccessedAt = lastAccessedAt
    )
}

fun ProductDomainModel.toEntity(): ProductEntity {
    return ProductEntity(
        barcode = barcode,
        name = name,
        brand = brand,
        imageUrl = imageUrl,
        nutriScore = nutriScore,
        novaGroup = novaGroup,
        ecoScore = ecoScore,
        ingredients = ingredients,
        categories = categories,
        allergens = allergens,
        additives = additives,
        energyKcal = energyKcal,
        fat = fat,
        saturatedFat = saturatedFat,
        carbohydrates = carbohydrates,
        sugars = sugars,
        fiber = fiber,
        proteins = proteins,
        salt = salt,
        dataSource = dataSource,
        lastAccessedAt = lastAccessedAt
    )
}
