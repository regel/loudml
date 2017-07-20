/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.elasticsearch.examples.nativescript.plugin;

import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.examples.nativescript.script.DeviceSearchScriptFactory;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.ScriptPlugin;
import org.elasticsearch.script.NativeScriptFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.elasticsearch.examples.nativescript.script.DeviceSearchScriptFactory.DEVICE_SEARCH_SCRIPT_DEFAULT_FIELD_NAME;

/**
 * This class is instantiated when Elasticsearch loads the plugin for the
 * first time. If you change the name of this plugin, make sure to update
 * src/main/resources/es-plugin.properties file that points to this class.
 */
public class BeatNativeScriptPlugin extends Plugin implements ScriptPlugin {

    private final Settings settings;

    public BeatNativeScriptPlugin(Settings settings) {
        this.settings = settings;
    }

    @Override
    public List<Setting<?>> getSettings() {
        return Collections.singletonList(DEVICE_SEARCH_SCRIPT_DEFAULT_FIELD_NAME);
    }

    @Override
    public List<NativeScriptFactory> getNativeScripts() {
        return Arrays.asList(
            new DeviceSearchScriptFactory(settings)
        );
    }
}
