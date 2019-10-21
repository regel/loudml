'use strict'

/**
 * Example for demonstrating hippie-swagger usage, including dereferencing
 *
 * Usage:  mocha example/index.js
 */

var SwaggerParser = require('swagger-parser')
var parser = new SwaggerParser()
var hippie = require('hippie-swagger')
var assert = require('chai').assert
var expect = require('chai').expect
var path = require('path')
var dereferencedSwagger
var swaggerFile = process.env.SWAGGER_FILE;
var targetUrl = process.env.TARGET_URL;

var options = {
  validateResponseSchema: true,
  validateParameterSchema: true,
  errorOnExtraParameters: true,
  errorOnExtraHeaderParameters: false
};


function api() {
  return hippie(dereferencedSwagger, options)
    .json()
    .base(targetUrl)
}


function assertVersion(body) {
  var version = body.version
  return assert.match(version, /^[0-9]+\.[0-9]+\.[0-9]+/, 'version regexp matches')
}


describe('Example of', function () {
  this.timeout(10000) // very large swagger files may take a few seconds to parse

  before(function (done) {
    // if using mocha, dereferencing can be performed prior during initialization via the delay flag:
    // https://mochajs.org/#delayed-root-suite
    parser.dereference(swaggerFile, function (err, api) {
      if (err) return done(err)
      dereferencedSwagger = api
      done()
    })
  })

  describe('correct usage', function () {
    it('works when the request matches the swagger file', function (done) {
      api()
        .get('/')
        .expectStatus(200)
        .expectHeader('content-type', 'application/json')
        .expect(function(res, body, next) {
          next(assertVersion(body));
        })
        .end(done)
    })
  })

  describe('things hippie-swagger will punish you for:', function () {
    it('validates paths', function (done) {
      try {
        api()
          .get('/undocumented-endpoint')
          .end(done)
      } catch (ex) {
        expect(ex.message).to.equal('Swagger spec does not define path: /undocumented-endpoint')
        done()
      }
    })
  })
})
