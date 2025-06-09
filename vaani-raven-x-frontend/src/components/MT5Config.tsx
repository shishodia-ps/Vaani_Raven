import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Settings, Wifi, WifiOff, Save, TestTube } from 'lucide-react';

interface MT5Credentials {
  server: string;
  login: string;
  password: string;
  investor?: string;
}

interface MT5ConfigProps {
  onCredentialsUpdate: (credentials: MT5Credentials) => void;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
}

const MT5Config: React.FC<MT5ConfigProps> = ({ onCredentialsUpdate, connectionStatus }) => {
  const [credentials, setCredentials] = useState<MT5Credentials>({
    server: 'MetaQuotes-Demo',
    login: '',
    password: '',
    investor: ''
  });

  const [isEditing, setIsEditing] = useState(false);
  const [isTesting, setIsTesting] = useState(false);

  const handleInputChange = (field: keyof MT5Credentials, value: string) => {
    setCredentials(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSaveCredentials = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/mt5/configure', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const result = await response.json();
      if (result.status === 'success') {
        onCredentialsUpdate(credentials);
        setIsEditing(false);
        alert('MT5 credentials saved successfully!');
      } else {
        alert(`Failed to save MT5 credentials: ${result.message}`);
      }
    } catch (error) {
      console.error('Error saving credentials:', error);
      alert('Error saving MT5 credentials');
    }
  };

  const handleTestConnection = async () => {
    setIsTesting(true);
    try {
      const response = await fetch('http://localhost:8000/api/mt5/test-connection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const result = await response.json();
      if (result.success) {
        alert('MT5 connection test successful!');
      } else {
        alert(`MT5 connection test failed: ${result.error || result.message}`);
      }
    } catch (error) {
      console.error('Error testing connection:', error);
      alert('Error testing MT5 connection');
    } finally {
      setIsTesting(false);
    }
  };

  const getConnectionStatusBadge = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Badge className="bg-green-600 text-white"><Wifi className="h-3 w-3 mr-1" />Connected</Badge>;
      case 'connecting':
        return <Badge className="bg-yellow-600 text-white"><Settings className="h-3 w-3 mr-1" />Connecting</Badge>;
      default:
        return <Badge className="bg-red-600 text-white"><WifiOff className="h-3 w-3 mr-1" />Disconnected</Badge>;
    }
  };

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <Settings className="h-5 w-5 text-blue-500" />
              <span>MT5 Configuration</span>
            </CardTitle>
            <CardDescription>Configure MetaTrader 5 connection for live data</CardDescription>
          </div>
          {getConnectionStatusBadge()}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {!isEditing ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-gray-400">Server:</div>
                <div className="font-semibold">{credentials.server || 'Not configured'}</div>
              </div>
              <div>
                <div className="text-gray-400">Login:</div>
                <div className="font-semibold">{credentials.login || 'Not configured'}</div>
              </div>
            </div>
            
            <div className="flex space-x-2">
              <Button 
                onClick={() => setIsEditing(true)}
                variant="outline"
                className="flex-1"
              >
                <Settings className="h-4 w-4 mr-2" />
                Configure
              </Button>
              
              {credentials.login && (
                <Button 
                  onClick={handleTestConnection}
                  disabled={isTesting}
                  variant="outline"
                  className="flex-1"
                >
                  <TestTube className="h-4 w-4 mr-2" />
                  {isTesting ? 'Testing...' : 'Test Connection'}
                </Button>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Server</label>
              <Input
                type="text"
                value={credentials.server}
                onChange={(e) => handleInputChange('server', e.target.value)}
                placeholder="MetaQuotes-Demo"
                className="bg-gray-700 border-gray-600"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Login</label>
              <Input
                type="text"
                value={credentials.login}
                onChange={(e) => handleInputChange('login', e.target.value)}
                placeholder="10006567513"
                className="bg-gray-700 border-gray-600"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Password</label>
              <Input
                type="password"
                value={credentials.password}
                onChange={(e) => handleInputChange('password', e.target.value)}
                placeholder="Enter your MT5 password"
                className="bg-gray-700 border-gray-600"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Investor Password (Optional)</label>
              <Input
                type="password"
                value={credentials.investor || ''}
                onChange={(e) => handleInputChange('investor', e.target.value)}
                placeholder="Enter investor password if available"
                className="bg-gray-700 border-gray-600"
              />
            </div>

            <div className="flex space-x-2">
              <Button 
                onClick={handleSaveCredentials}
                className="flex-1 bg-blue-600 hover:bg-blue-700"
              >
                <Save className="h-4 w-4 mr-2" />
                Save Configuration
              </Button>
              
              <Button 
                onClick={() => setIsEditing(false)}
                variant="outline"
                className="flex-1"
              >
                Cancel
              </Button>
            </div>
          </div>
        )}

        <div className="text-xs text-gray-400 bg-gray-700 p-3 rounded">
          <strong>Note:</strong> Your MT5 credentials are securely transmitted to the backend for establishing live data connection. 
          The system will automatically switch from mock data to live EUR/USD feeds once connected.
        </div>
      </CardContent>
    </Card>
  );
};

export default MT5Config;
