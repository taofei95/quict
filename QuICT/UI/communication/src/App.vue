<template>
  <Home></Home>
</template>

<script>
import Home from "./components/Home.vue";
export default {
  name: "App",
  components: {
    Home,
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

.VisPro {
  height: 100vh;
}
@font-face {
  font-family: "Segoe UI Symbol";
  src: local("Segoe UI Symbol"), url(./assets/seguisym.ttf) format("truetype");
}
</style>
