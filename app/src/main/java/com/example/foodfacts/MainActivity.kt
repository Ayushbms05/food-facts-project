package com.example.foodfacts

import android.os.Bundle
import android.view.View
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupWithNavController
import com.example.foodfacts.databinding.ActivityMainBinding
import com.google.android.material.appbar.AppBarLayout

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.main) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, 0)
            insets
        }

        val navHostFragment = supportFragmentManager
            .findFragmentById(R.id.navHostFragment) as NavHostFragment
        val navController = navHostFragment.navController

        val appBarConfiguration = AppBarConfiguration(
            setOf(
                R.id.homeFragment,
                R.id.scanFragment,
                R.id.searchFragment,
                R.id.historyFragment,
                R.id.compareFragment
            )
        )

        binding.toolbar.setupWithNavController(navController, appBarConfiguration)
        binding.bottomNav.setupWithNavController(navController)
        
        val appBarLayout = binding.toolbar.parent as AppBarLayout

        navController.addOnDestinationChangedListener { _, destination, _ ->
            when (destination.id) {
                R.id.splashFragment, R.id.loginFragment -> {
                    appBarLayout.visibility = View.GONE
                    binding.bottomNav.visibility = View.GONE
                }
                else -> {
                    appBarLayout.visibility = View.VISIBLE
                    binding.bottomNav.visibility = View.VISIBLE
                }
            }
        }
    }
}
