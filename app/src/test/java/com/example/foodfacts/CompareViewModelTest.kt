package com.example.foodfacts

import com.example.foodfacts.data.local.ProductDao
import com.example.foodfacts.data.local.ScanEventDao
import com.example.foodfacts.data.model.ComparisonWinner
import com.example.foodfacts.data.model.ProductDomainModel
import com.example.foodfacts.data.repository.ProductRepository
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.mockito.kotlin.mock

class CompareViewModelTest {

    private lateinit var productDao: ProductDao
    private lateinit var scanEventDao: ScanEventDao
    private lateinit var repository: ProductRepository

    @Before
    fun setUp() {
        productDao = mock()
        scanEventDao = mock()
        repository = ProductRepository(productDao, scanEventDao, mock())
    }

    @Test
    fun `compareProducts produces ties for identical products`() {
        val product = ProductDomainModel(
            barcode = "1",
            name = "Water",
            brand = "Pure",
            imageUrl = null,
            nutriScore = "A",
            novaGroup = 1,
            ecoScore = "A",
            ingredients = "Water",
            categories = "Beverages",
            allergens = null,
            energyKcal = 0.0,
            fat = 0.0,
            saturatedFat = 0.0,
            carbohydrates = 0.0,
            sugars = 0.0,
            fiber = 0.0,
            proteins = 0.0,
            salt = 0.0
        )

        val result = repository.compareProducts(product, product)

        assertEquals(ComparisonWinner.TIE, result.recommendedProduct)
        assertEquals("Both products have comparable overall nutritional performance.", result.summary)
    }
}
