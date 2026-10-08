plugins {
    kotlin("jvm") version "2.4.10"
    kotlin("plugin.serialization") version "2.4.10"
    application
}
group = "org.hyperl"
version = "0.1.0-alpha.6"
repositories { mavenCentral() }
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
java { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
dependencies {
    implementation("com.formdev:flatlaf:3.7.2:no-natives")
    implementation("com.fifesoft:rsyntaxtextarea:4.0.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.0")
    testImplementation("junit:junit:4.13.2")
}
application { mainClass.set("org.hyperl.MainKt"); applicationName = "hyperl" }
distributions {
    main {
        contents {
            from("README.md", "LICENSE", "NOTICE")
            from("examples") { into("examples") }
            from("contracts") { into("contracts") }
            from("python") { into("python"); exclude("__pycache__/**", "**/__pycache__/**", "*.pyc", "**/*.egg-info/**", "build/**") }
            from("docs") { into("docs") }
            from("native") { into("sdk/native"); exclude("hyperl-opencl", "hyperl-opencl.exe", "hyperl-metal") }
            from("CMakeLists.txt") { into("sdk") }
            from("scripts") { into("scripts"); exclude("__pycache__/**", "*.pyc") }
        }
    }
}
tasks.test { systemProperty("java.awt.headless", "true") }
