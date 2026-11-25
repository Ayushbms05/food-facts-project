package com.example.foodfacts

import com.example.foodfacts.data.api.OpenFoodFactsApi
import com.example.foodfacts.data.api.model.ApiNutriments
import com.example.foodfacts.data.api.model.ApiProduct
import com.example.foodfacts.data.api.model.ProductResponse
import com.example.foodfacts.data.local.ProductDao
import com.example.foodfacts.data.local.ProductEntity
import com.example.foodfacts.data.local.ScanEventDao
import com.example.foodfacts.data.model.ComparisonWinner
import com.example.foodfacts.data.model.ProductDomainModel
import com.example.foodfacts.data.model.Resource
import com.example.foodfacts.data.model.ScanSource
import com.example.foodfacts.data.repository.ProductRepository
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.mockito.kotlin.any
import org.mockito.kotlin.mock
import org.mockito.kotlin.verify
import org.mockito.kotlin.whenever
import retrofit2.Response

@OptIn(ExperimentalCoroutinesApi::class)
class ProductRepositoryTest {

    private lateinit var productDao: ProductDao
    private lateinit var scanEventDao: ScanEventDao
    private lateinit var api: OpenFoodFactsApi
    private lateinit var repository: ProductRepository

    @Before
    fun setUp() {
        productDao = mock()
        scanEventDao = mock()
        api = mock()
        repository = ProductRepository(productDao, scanEventDao, api)
    }

    @Test
    fun `getProductByBarcode returns Success when API responds with product`() = runTest {
        val barcode = "3017620422003"
        val apiProduct = ApiProduct(
            code = barcode,
            productName = "Nutella",
            brands = "Ferrero",
            nutriscoreGrade = "E",
            novaGroup = 4,
            nutriments = ApiNutriments(energyKcal100g = 539.0, sugars100g = 56.3)
        )
        val response = ProductResponse(code = barcode, status = 1, product = apiProduct)

        whenever(api.getProductByBarcode(barcode)).thenReturn(Response.success(response))

        val result = repository.getProductByBarcode(barcode, ScanSource.BARCODE)

        assertTrue(result is Resource.Success)
        val product = (result as Resource.Success).data
        assertEquals("Nutella", product.name)
        assertEquals("Ferrero", product.brand)
        assertEquals("E", product.nutriScore)
        assertEquals(4, product.novaGroup)
        verify(productDao).insertOrUpdate(any())
        verify(scanEventDao).insertEvent(any())
    }

    @Test
    fun `getProductByBarcode returns cached product when network request fails`() = runTest {
        val barcode = "123456"
        val cachedEntity = ProductEntity(
            barcode = barcode,
            name = "Cached Oats",
            brand = "Quaker",
            imageUrl = null,
            nutriScore = "A",
            novaGroup = 1,
            ecoScore = "A",
            ingredients = "Oats",
            categories = "Cereals",
            allergens = null,
            additives = null,
            energyKcal = 389.0,
            fat = 6.9,
            saturatedFat = 1.2,
            carbohydrates = 66.0,
            sugars = 1.0,
            fiber = 10.0,
            proteins = 17.0,
            salt = 0.01,
            dataSource = "Room Cache",
            lastAccessedAt = System.currentTimeMillis()
        )

        whenever(productDao.getProductByBarcode(barcode)).thenReturn(cachedEntity)
        whenever(api.getProductByBarcode(barcode)).thenThrow(RuntimeException("Network offline"))

        val result = repository.getProductByBarcode(barcode)

        assertTrue(result is Resource.Error)
        val errorResult = result as Resource.Error
        assertNotNull(errorResult.cachedData)
        assertEquals("Cached Oats", errorResult.cachedData?.name)
    }

    @Test
    fun `compareProducts correctly determines overall healthier product`() {
        val healthyProduct = ProductDomainModel(
            barcode = "1",
            name = "Organic Oats",
            brand = "Nature",
            imageUrl = null,
            nutriScore = "A",
            novaGroup = 1,
            ecoScore = "A",
            ingredients = "Whole Oats",
            categories = "Grains",
            allergens = null,
            energyKcal = 350.0,
            fat = 5.0,
            saturatedFat = 1.0,
            carbohydrates = 60.0,
            sugars = 1.0,
            fiber = 10.0,
            proteins = 14.0,
            salt = 0.01
        )

        val junkProduct = ProductDomainModel(
            barcode = "2",
            name = "Sugar Candy",
            brand = "SweetCo",
            imageUrl = null,
            nutriScore = "E",
            novaGroup = 4,
            ecoScore = "E",
            ingredients = "Sugar, Color",
            categories = "Sweets",
            allergens = null,
            energyKcal = 550.0,
            fat = 25.0,
            saturatedFat = 15.0,
            carbohydrates = 80.0,
            sugars = 70.0,
            fiber = 0.0,
            proteins = 1.0,
            salt = 0.5
        )

        val comparison = repository.compareProducts(healthyProduct, junkProduct)

        assertEquals(ComparisonWinner.PRODUCT_A, comparison.recommendedProduct)
        assertTrue(comparison.summary.contains("Organic Oats is recommended overall"))
    }
}
