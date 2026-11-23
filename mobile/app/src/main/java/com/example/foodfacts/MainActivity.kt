
package com.example.foodfacts

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.google.mlkit.vision.codescanner.GmsBarcodeScanning

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                FoodFactsApp()
            }
        }
    }
}

@Composable
fun FoodFactsApp() {
    val context = LocalContext.current
    val scanner = remember(context) {
        GmsBarcodeScanning.getClient(context)
    }

    var showProduct by remember {
        mutableStateOf(false)
    }

    var scannedBarcode by remember {
        mutableStateOf<String?>(null)
    }

    var scanMessage by remember {
        mutableStateOf<String?>(null)
    }

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            if (!showProduct) {
                Text(
                    text = "Food Facts",
                    style = MaterialTheme.typography.headlineLarge
                )

                Spacer(modifier = Modifier.height(12.dp))

                Text(
                    text = "Understand what's inside your food.",
                    textAlign = TextAlign.Center
                )

                Spacer(modifier = Modifier.height(32.dp))

                Button(
                    onClick = {
                        scanMessage = null

                        scanner.startScan()
                            .addOnSuccessListener { barcode ->
                                scannedBarcode = barcode.rawValue
                                scanMessage = "Barcode scanned successfully!"
                            }
                            .addOnCanceledListener {
                                scanMessage = "Scanning cancelled"
                            }
                            .addOnFailureListener { error ->
                                scanMessage = "Scan failed: ${error.message}"
                            }
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Scan Food Barcode")
                }

                Spacer(modifier = Modifier.height(16.dp))

                OutlinedButton(
                    onClick = { showProduct = true },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("View Sample Product")
                }

                Spacer(modifier = Modifier.height(24.dp))

                scanMessage?.let {
                    Text(text = it)
                }

                scannedBarcode?.let {
                    Spacer(modifier = Modifier.height(16.dp))

                    Card(modifier = Modifier.fillMaxWidth()) {
                        Column(
                            modifier = Modifier.padding(20.dp)
                        ) {
                            Text("Scanned Barcode")
                            Text(
                                text = it,
                                style = MaterialTheme.typography.titleLarge
                            )
                            Text(
                                "Product lookup not connected yet.",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            } else {
                Text(
                    "Product Details",
                    style = MaterialTheme.typography.headlineMedium
                )

                Spacer(modifier = Modifier.height(20.dp))

                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(20.dp)) {
                        Text(
                            "Demo Biscuit",
                            style = MaterialTheme.typography.titleLarge
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        Text("Ingredients: Wheat flour, sugar, vegetable oil")

                        Spacer(modifier = Modifier.height(12.dp))

                        Text("NOVA prediction: Not connected yet")

                        Text("Source: Mock data")
                    }
                }

                Spacer(modifier = Modifier.height(20.dp))

                OutlinedButton(
                    onClick = { showProduct = false }
                ) {
                    Text("Back to Home")
                }
            }
        }
    }
}

