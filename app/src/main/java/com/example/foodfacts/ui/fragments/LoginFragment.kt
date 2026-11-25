package com.example.foodfacts.ui.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import coil.load
import com.example.foodfacts.R
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText

class LoginFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_login, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        val topImage = view.findViewById<ImageView>(R.id.ivLoginTop)
        topImage.load("https://images.unsplash.com/photo-1498837167922-ddd27525d352?q=80&w=1080&auto=format&fit=crop") {
            crossfade(true)
        }

        val btnLogin = view.findViewById<MaterialButton>(R.id.btnLogin)
        val etEmail = view.findViewById<TextInputEditText>(R.id.etEmail)
        val etPassword = view.findViewById<TextInputEditText>(R.id.etPassword)

        btnLogin.setOnClickListener {
            val email = etEmail.text.toString().trim()
            val password = etPassword.text.toString().trim()
            
            if (email.isNotEmpty() && password.isNotEmpty()) {
                findNavController().navigate(R.id.action_loginFragment_to_homeFragment)
            } else {
                Toast.makeText(context, "Please enter email and password", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
