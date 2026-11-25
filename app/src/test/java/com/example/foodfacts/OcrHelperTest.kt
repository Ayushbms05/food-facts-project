package com.example.foodfacts

import com.example.foodfacts.util.OcrHelper
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OcrHelperTest {

    @Test
    fun `detectAllergens identifies common allergens in raw label text`() {
        val labelText = "Ingredients: Sugar, Skimmed Milk Powder, Wheat Flour, Hazelnuts, Soy Lecithin, Salt."

        val allergens = OcrHelper.detectAllergens(labelText)

        assertTrue(allergens.contains("Milk"))
        assertTrue(allergens.contains("Wheat"))
        assertTrue(allergens.contains("Soy"))
    }

    @Test
    fun `detectAllergens returns empty list when no allergens present`() {
        val labelText = "Ingredients: 100% Pure Spring Water."

        val allergens = OcrHelper.detectAllergens(labelText)

        assertTrue(allergens.isEmpty())
    }
}
