package com.example.foodfacts.ui.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import coil.load
import com.example.foodfacts.R
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class SplashFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_splash, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        val bgImage = view.findViewById<ImageView>(R.id.ivSplashBg)
        // Load a beautiful healthy food background image
        bgImage.load("https://images.unsplash.com/photo-1490645935967-10de6ba17061?q=80&w=1080&auto=format&fit=crop") {
            crossfade(true)
        }

        lifecycleScope.launch {
            delay(2500)
            if (isAdded) {
                findNavController().navigate(R.id.action_splashFragment_to_loginFragment)
            }
        }
    }
}
