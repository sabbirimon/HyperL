plugins {
    id("com.android.application") version "9.2.1"
    id("org.jetbrains.kotlin.plugin.serialization") version "2.4.10"
}
android {
    namespace = "org.hyperl.probe"
    compileSdk = 37
    defaultConfig { applicationId = "org.hyperl.probe"; minSdk = 26; targetSdk = 36; versionCode = 1; versionName = "0.1-probe" }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    sourceSets.getByName("main").assets.directories.add(layout.buildDirectory.dir("notice-assets").get().asFile.absolutePath)
    sourceSets.getByName("main").kotlin.directories.add(layout.buildDirectory.dir("portable-source").get().asFile.absolutePath)
}
val preparePortableSource by tasks.registering(Sync::class) {
    from("../../src/main/kotlin") {
        include("org/hyperl/HyperL.kt", "org/hyperl/HyperLAdmission.kt", "org/hyperl/HyperLCodec.kt", "org/hyperl/HyperLContract.kt", "org/hyperl/HyperLLibrary.kt", "org/hyperl/PreciseReduction.kt")
    }
    into(layout.buildDirectory.dir("portable-source"))
}
val prepareNotices by tasks.registering(Copy::class) {
    from("../../LICENSE", "../../NOTICE", "../../docs/LICENSE_HISTORY.md")
    from("../../docs/licenses/Apache-2.0.txt")
    into(layout.buildDirectory.dir("notice-assets"))
}
tasks.named("preBuild").configure { dependsOn(prepareNotices,preparePortableSource) }
dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.0")
}
