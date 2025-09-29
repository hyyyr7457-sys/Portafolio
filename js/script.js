// script.js — Manejo básico de interacción
// Captura el envío del formulario de contacto y muestra un mensaje en consola.

document.addEventListener('DOMContentLoaded', function onReady() {
  var form = document.getElementById('contactForm');
  if (!form) return;

  form.addEventListener('submit', function handleSubmit(event) {
    event.preventDefault();

    // Lectura simple de campos
    var nombre = document.getElementById('nombre')?.value || '';
    var correo = document.getElementById('correo')?.value || '';
    var mensaje = document.getElementById('mensaje')?.value || '';

    // Validación mínima de ejemplo
    if (!nombre || !correo || !mensaje) {
      console.warn('Por favor completa todos los campos antes de enviar.');
      return;
    }

    console.log('Formulario enviado correctamente');

    // Feedback visual ligero
    form.reset();
    form.querySelector('button[type="submit"]').textContent = '¡Enviado!';
    setTimeout(function restore() {
      form.querySelector('button[type="submit"]').textContent = 'Enviar';
    }, 1600);
  });
});


